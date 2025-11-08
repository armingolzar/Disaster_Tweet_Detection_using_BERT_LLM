import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model
from transformers import TFBertModel
import src.config as config

class Disaster_Detection_BERT_Model():

    def __init__(self):
        pass

    
    @staticmethod
    def building_network():

        bert_model = TFBertModel.from_pretrained(config.BERT_MODEL_NAME)

        # bert_model.trainable = True
        # for var in bert_model.trainable_variables: 
        #     print(var.name)                        # this shows the full structure of the model and you can check what you want to freeze and what you dont 

        # for i, layer in enumerate(bert_model.bert.encoder.layer):
        #     if i < 11:
        #         layer.trainable = False
        #     else:
        #         layer.trainable = True  

        # bert_model.bert.embeddings.trainable = False        
        # bert_model.bert.pooler.trainable = True

        bert_model.trainable = False

            

        for i, enc_layer in enumerate(bert_model.bert.encoder.layer):
            print(f"Encoder layer {i} trainable: {enc_layer.trainable}")
        print("Embeddings trainable:", bert_model.bert.embeddings.trainable)
        print("Pooler trainable:", bert_model.bert.pooler.trainable)

        input_tokens = Input(shape=(config.SEN_LENGTH,), dtype=tf.int64, name="input_tokens")
        input_masks = Input(shape=(config.SEN_LENGTH,), dtype=tf.int64, name="input_masks")

        bert_embedding = bert_model(input_tokens, attention_mask=input_masks) # with [1] at the end of this line we could get cls_embedding
        # cls_embedding = bert_embedding.pooler_output
        cls_embedding = bert_embedding[0][:, 0, :] # this is the raw cls embedding after multi-head-self-attention but pooler has an extra dense layer on this raw embeding

        drop1 = Dropout(0.2, name="drop1")(cls_embedding)
        dense1 = Dense(256, activation="relu", name="dense1")(drop1)
        norm1 = LayerNormalization(name="norm1")(dense1)
        drop2 = Dropout(0.2, name="drop2")(norm1)
        dense2 = Dense(128, activation="relu", name="dense2")(drop2)
        norm2 = LayerNormalization(name="norm2")(dense2)
        drop3 = Dropout(0.2, name="drop3")(norm2)
        dense3 = Dense(64, activation="relu", name="dense3")(drop3)
        norm3 = LayerNormalization(name="norm3")(dense3)
        drop4 = Dropout(0.2, name="drop4")(norm3)
        dense4 = Dense(32, activation="relu", name="dense4")(drop4)
        # norm4 = LayerNormalization(name="norm4")(dense4)
        drop5 = Dropout(0.2, name="drop5")(dense4)
        # dense5 = Dense(16, activation="relu", name="dense5")(drop5)

        output = Dense(1, activation="sigmoid", name="output")(drop5)

        disaster_detection_model = Model(inputs=[input_tokens, input_masks], outputs=output)
        disaster_detection_model.summary()

        return disaster_detection_model
    


    @staticmethod
    def compil_model(model, optimizer="Adam", learning_rate=1e-5): # for feature extraction use 1e-3 but for fine-tuning we need smaller updates 1e-5

        if isinstance(optimizer, str):
            optimizer = tf.keras.optimizers.get(optimizer)
            optimizer.learning_rate = learning_rate

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model
    

    @staticmethod
    def training_model(model, train_data, epochs, validation_data):

        history = model.fit(train_data, epochs=epochs, validation_data=validation_data)
    
        return model, history
    





