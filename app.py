import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from keras.models import Model # type: ignore
from keras.layers import Dense, Input # type: ignore
from keras.optimizers import Adam # type: ignore
from sklearn.model_selection import train_test_split
import os

# Get the absolute path to the current script
path = os.path.abspath(__file__)

# Set the working directory to the script's directory
os.chdir(os.path.dirname(path))

# Specify the path to the model folder relative to the script's directory
file = "Multiple CSV"

file_paths = {
    "ushape dataset": os.path.join(file, '1.ushape.csv'),
    "concerticcir1 dataset": os.path.join(file, '2.concerticcir1.csv'),
    "concertriccir2 dataset": os.path.join(file, '3.concertriccir2.csv'),
    "linearsep dataset": os.path.join(file, '4.linearsep.csv'),
    "outlier dataset": os.path.join(file, '5.outlier.csv'),
    "overlap dataset": os.path.join(file, '6.overlap.csv'),
    "xor dataset": os.path.join(file, '7.xor.csv'),
    "twospirals dataset": os.path.join(file, '8.twospirals.csv')
}

st.header('Tensorflow Playground')

# Function to load datasets
def load_data(file_path):
    return pd.read_csv(file_path, header=None)

def app():
    st.sidebar.title("Settings")
    
    dataset_choice = st.sidebar.selectbox("Choose dataset", list(file_paths.keys()))
    dataset_path = file_paths[dataset_choice]
    
    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 1, 7, 1)
    epochs = st.sidebar.slider("Number of Epochs", 100, 1000, 100)
    learning_rate = st.sidebar.number_input('Learning Rate', value=0.001, format="%.6f")
    
    hidden_layer_configs = []

    for i in range(num_hidden_layers):
        st.sidebar.markdown(f"Hidden Layer {i+1}")
        units = st.sidebar.slider(f"Number of units in Hidden Layer {i+1}", 1, 10, 1)
        activation = st.sidebar.selectbox(f"Activation function for Hidden Layer {i+1}", ['tanh', 'sigmoid', "linear"], key=f"activation_{i}")
        hidden_layer_configs.append((units, activation))
    
    if st.sidebar.button("Submit"):
        # Load dataset
        dataset = load_data(dataset_path)

        # Splitting dataset
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values.astype(np.int_)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=23)
        
        # Build model
        input_layer = Input(shape=(2,))
        x = input_layer
        for units, activation in hidden_layer_configs:
            x = Dense(units=units, activation=activation)(x)
        output_layer = Dense(units=1, activation='sigmoid')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), verbose=0)
        
        # Plot training and testing loss
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Testing Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)
        
        # Plot the output layer decision region
        fig, ax = plt.subplots()
        plot_decision_regions(X, Y, clf=model, ax=ax)
        st.pyplot(fig)
        
        # Plot decision regions for each hidden layer neuron
        hidden_layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Dense)]
        
        for layer_num, layer_output in enumerate(hidden_layer_outputs[:-1]):  # Exclude the output layer
            num_neurons = layer_output.shape[1]
            cols = st.columns(3)  # Create 3 columns for the plots
            for neuron_num in range(num_neurons):
                neuron_model = Model(inputs=model.input, outputs=layer_output[:, neuron_num])
                col = cols[neuron_num % 3]  # Cycle through the columns
                with col:
                    st.write(f"Decision Region for Neuron {neuron_num+1} in Hidden Layer {layer_num+1}")
                    fig, ax = plt.subplots()
                    plot_decision_regions(X, Y, clf=neuron_model, ax=ax)
                    st.pyplot(fig)

if __name__ == "__main__":
    app()
