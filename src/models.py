from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping


def build_lstm(input_shape, units=64, dropout=0.2):
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units=units, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_bilstm(input_shape, units=64, dropout=0.2):
    model = Sequential([
        Bidirectional(LSTM(units=units, return_sequences=True), input_shape=input_shape),
        Dropout(dropout),
        Bidirectional(LSTM(units=units, return_sequences=False)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def build_gru(input_shape, units=64, dropout=0.2):
    model = Sequential([
        GRU(units=units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        GRU(units=units, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_model(model, X_train, y_train, epochs=100, batch_size=16, validation_split=0.2):
    """
    Train a model with EarlyStopping.
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=False,
        callbacks=[early_stop],
        verbose=1
    )
    return history
