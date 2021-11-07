import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix


def build_model(x_train, y_train, x_test, y_test, encoder):
    model = Sequential()
    model.add(Dense(250, input_dim=400, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, validation_split=0.3, epochs=2, batch_size=100)

    labels = model.predict(x_test)
    predictions = encoder.inverse_transform([0 if l[0] >= 0.5 else 1 for l in labels])

    tr_y_test = encoder.inverse_transform([0 if l[0] >= 0.5 else 1 for l in y_test])
    accuracy = accuracy_score(tr_y_test, predictions)
    conf_matrix_df = pd.DataFrame(confusion_matrix(tr_y_test, predictions),
                                  columns=encoder.classes_, index=encoder.classes_)

    test_info = {'accuracy': accuracy, 'conf_matrix_df': conf_matrix_df}

    return model, test_info
