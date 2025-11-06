import tensorflow as tf
import src.config as config
from src.data_loader import preparing_bert_training_datasets
from src.model import Disaster_Detection_BERT_Model
from src.utils import training_plot


def main():

    train_dataset, val_dataset = preparing_bert_training_datasets(config.TRAIN_PATH)
    print("✅ Datasets ready:")
    print(f"Train batches: {len(train_dataset)}, Validation batches: {len(val_dataset)}\n")


    disaster_detection_class = Disaster_Detection_BERT_Model()
    disaster_detection_model = disaster_detection_class.building_network()
    compiled_disaster_detection_model = disaster_detection_class.compil_model(disaster_detection_model)

    print("⏳ Training started...")
    trained_disaster_detection_model, history = disaster_detection_class.training_model(model=compiled_disaster_detection_model, train_data=train_dataset,
                                                                                        epochs=config.EPOCHS, validation_data=val_dataset)
    
    print("✅ Training completed!")
    trained_disaster_detection_model.save(config.MODEL_PATH)
    print(f"💾 Model saved at {config.MODEL_PATH}")

    training_plot(history)
    print(f"📊 Training metrics plot saved at {config.PLOT_PATH}")


if __name__ == "__main__":
    main()

