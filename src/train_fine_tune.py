import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os
from src.data_loader import preparing_bert_training_datasets
from src.utils import training_plot
import src.config as config 

# --- CONFIG ---
BERT_NAME = "bert-base-uncased"
SEQ_LEN = 100
BATCH_SIZE = 64
STAGE1_EPOCHS = 3            # BERT + dense
STAGE2_EXTRA_EPOCHS = 27     # dense-only
TOTAL_EPOCHS = STAGE1_EPOCHS + STAGE2_EXTRA_EPOCHS

LR_BERT = 2e-5
LR_DENSE = 1e-4
CLIP_NORM = 1.0

SAVE_PATH_MODELS = "..\\models\\bert_disaster_classifier_best_fine_tune"  # path to save best model
SAVE_PATH_PLOTS = "..\\assets\\fine_tuning_training_curve.png"

# --- Build model ---
bert_model = TFBertModel.from_pretrained(BERT_NAME)
bert_model.trainable = True

# Freeze embeddings & keep last 2 encoder layers trainable
for i, L in enumerate(bert_model.bert.encoder.layer):
    L.trainable = (i >= 10)
bert_model.bert.embeddings.trainable = False
bert_model.bert.pooler.trainable = True

# Head
input_tokens = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="input_tokens")
input_masks  = Input(shape=(SEQ_LEN,), dtype=tf.int32, name="input_masks")

bert_outputs = bert_model(input_tokens, attention_mask=input_masks)
cls = bert_outputs[0][:, 0, :]  # CLS token

x = tf.keras.layers.Dropout(0.2)(cls)
x = tf.keras.layers.Dense(256, activation="relu", name="dense1")(x)
x = tf.keras.layers.LayerNormalization(name="norm1")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation="relu", name="dense2")(x)
x = tf.keras.layers.LayerNormalization(name="norm2")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(16, activation="relu", name="dense3")(x)
output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=[input_tokens, input_masks], outputs=output)

# --- Identify variable groups ---
bert_vars = [v for v in model.trainable_variables if "/bert/" in v.name or v.name.startswith("bert/") or "tf_bert_model/bert" in v.name]
bert_var_ids = {id(v) for v in bert_vars}
dense_vars = [v for v in model.trainable_variables if id(v) not in bert_var_ids]

print(f"bert_vars: {len(bert_vars)} variables, dense_vars: {len(dense_vars)} variables")

# --- Optimizers ---
opt_bert  = tf.keras.optimizers.Adam(learning_rate=LR_BERT)
opt_dense = tf.keras.optimizers.Adam(learning_rate=LR_DENSE)

# Loss and metrics
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_acc  = tf.keras.metrics.BinaryAccuracy(name="train_acc")
val_loss   = tf.keras.metrics.Mean(name="val_loss")
val_acc    = tf.keras.metrics.BinaryAccuracy(name="val_acc")

# --- Helper: apply gradients with clipping ---
def apply_grads(optimizer, grads, vars_):
    grads, _ = tf.clip_by_global_norm(grads, CLIP_NORM)
    optimizer.apply_gradients(zip(grads, vars_))

# --- Training step ---
@tf.function
def train_step(batch_inputs, batch_labels, train_bert=True):
    with tf.GradientTape() as tape:
        preds = model(batch_inputs, training=True)
        loss = loss_fn(batch_labels, preds)

    grads = tape.gradient(loss, model.trainable_variables)
    var_to_grad = {v.ref(): g for v, g in zip(model.trainable_variables, grads)}
    bert_grads = [var_to_grad.get(v.ref()) for v in bert_vars]
    dense_grads = [var_to_grad.get(v.ref()) for v in dense_vars]

    if train_bert:
        bert_grads_filtered = [g for g in bert_grads if g is not None]
        bert_vars_filtered  = [v for v, g in zip(bert_vars, bert_grads) if g is not None]
        if bert_grads_filtered:
            apply_grads(opt_bert, bert_grads_filtered, bert_vars_filtered)

    dense_grads_filtered = [g for g in dense_grads if g is not None]
    dense_vars_filtered  = [v for v, g in zip(dense_vars, dense_grads) if g is not None]
    if dense_grads_filtered:
        apply_grads(opt_dense, dense_grads_filtered, dense_vars_filtered)

    train_loss.update_state(loss)
    train_acc.update_state(batch_labels, preds)

@tf.function
def valid_step(batch_inputs, batch_labels):
    preds = model(batch_inputs, training=False)
    loss = loss_fn(batch_labels, preds)
    val_loss.update_state(loss)
    val_acc.update_state(batch_labels, preds)

# --- Training orchestration ---
def train(train_dataset, val_dataset):
    best_val_loss = 1e9  # Initialize very large value

    # For plotting history
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    # --- Stage 1: train BERT + dense ---
    for epoch in range(STAGE1_EPOCHS):
        train_loss.reset_states(); train_acc.reset_states()
        val_loss.reset_states(); val_acc.reset_states()

        for batch in train_dataset:
            batch_inputs = {"input_tokens": batch[0][0], "input_masks": batch[0][1]} if isinstance(batch[0], tuple) else {"input_tokens": batch[0]["input_tokens"], "input_masks": batch[0]["input_masks"]}
            batch_labels = batch[1]
            train_step(batch_inputs, batch_labels, train_bert=True)

        for vb in val_dataset:
            v_inputs = {"input_tokens": vb[0][0], "input_masks": vb[0][1]} if isinstance(vb[0], tuple) else {"input_tokens": vb[0]["input_tokens"], "input_masks": vb[0]["input_masks"]}
            valid_step(v_inputs, vb[1])

        # Record metrics
        train_loss_history.append(float(train_loss.result()))
        val_loss_history.append(float(val_loss.result()))
        train_acc_history.append(float(train_acc.result()))
        val_acc_history.append(float(val_acc.result()))

        print(f"Stage1 Epoch {epoch+1}/{STAGE1_EPOCHS} - loss: {train_loss.result():.4f}, acc: {train_acc.result():.4f}, val_loss: {val_loss.result():.4f}, val_acc: {val_acc.result():.4f}")

        # --- Save best model ---
        if val_loss.result() < best_val_loss:
            best_val_loss = val_loss.result()
            model.save(SAVE_PATH_MODELS, save_format="tf")
            print(f"✅ Best model saved (Stage1) with val_loss = {best_val_loss:.4f}")

    # --- Stage 2: freeze BERT, train dense-only ---
    print("Freezing BERT for stage2 (dense-only training)...")
    bert_model.trainable = False

    for epoch in range(STAGE2_EXTRA_EPOCHS):
        train_loss.reset_states(); train_acc.reset_states()
        val_loss.reset_states(); val_acc.reset_states()

        for batch in train_dataset:
            batch_inputs = {"input_tokens": batch[0][0], "input_masks": batch[0][1]} if isinstance(batch[0], tuple) else {"input_tokens": batch[0]["input_tokens"], "input_masks": batch[0]["input_masks"]}
            batch_labels = batch[1]
            train_step(batch_inputs, batch_labels, train_bert=False)

        for vb in val_dataset:
            v_inputs = {"input_tokens": vb[0][0], "input_masks": vb[0][1]} if isinstance(vb[0], tuple) else {"input_tokens": vb[0]["input_tokens"], "input_masks": vb[0]["input_masks"]}
            valid_step(v_inputs, vb[1])

        # Record metrics
        train_loss_history.append(float(train_loss.result()))
        val_loss_history.append(float(val_loss.result()))
        train_acc_history.append(float(train_acc.result()))
        val_acc_history.append(float(val_acc.result()))

        print(f"Stage2 Epoch {epoch+1}/{STAGE2_EXTRA_EPOCHS} - loss: {train_loss.result():.4f}, acc: {train_acc.result():.4f}, val_loss: {val_loss.result():.4f}, val_acc: {val_acc.result():.4f}")

        # --- Save best model ---
        if val_loss.result() < best_val_loss:
            best_val_loss = val_loss.result()
            model.save(SAVE_PATH_MODELS, save_format="tf")
            print(f"✅ Best model saved (Stage2) with val_loss = {best_val_loss:.4f}")

    print("Training complete.")
    print(f"📦 Best model stored at: {os.path.abspath(SAVE_PATH_MODELS)}")

    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "train_acc": train_acc_history,
        "val_acc": val_acc_history}
    
    return history

train_dataset, val_dataset = preparing_bert_training_datasets(config.TRAIN_PATH)
history = train(train_dataset, val_dataset)
training_plot(history)



