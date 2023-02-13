from pretrained_bert import build_classifier_model, compile_classifier_model
from tensorflow.keras.callbacks import ModelCheckpoint, Tensorboard
import argparse
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', action="store_true", default=1e-5)
    parser.add_argument('-e', '--epochs', action="store_true", default=10)
    parser.add_argument('-cp', '--checkpoint', action="store_true", default = '../Model')
    parser.add_argument('-tp', '--tensorboard', action="store_true", default = '../tensorboard')
    return parser
    

if __name__ == "__main__":
    
    parser = parse_args()
    args = parser.parse_args()
    train_ds, val_ds = None, None
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    num_train_steps = steps_per_epoch * args.epochs
    num_warmup_steps = int(0.1*num_train_steps)

    checkpoint_filepath = '../Model'
    tensorboard_filepath = '../tensorboard'
    classifier_model = build_classifier_model()
    classifier_model = compile_classifier_model(classifier_model, args.learning_rate, num_train_steps, num_warmup_steps)
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    model_tensorboard_callback = Tensorboard(log_dir='../tensorboard')
    

    history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                                  epochs=args.epochs,
                                  callbacks = [model_checkpoint_callback, model_tensorboard_callback])
    
