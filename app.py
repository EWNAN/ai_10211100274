import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.feature_extraction.text import TfidfVectorizer

# TensorFlow is used in the Neural Network module
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")

# For PDF text extraction
import PyPDF2
import io

def extract_text_from_pdf(file_buffer):
    try:
        reader = PyPDF2.PdfReader(file_buffer)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {e}")
        return ""

# ----------------------------
# Regression Module Function
# ----------------------------
def regression_module():
    st.header("Regression Model")
    
    # File upload and preview
    uploaded_file = st.file_uploader("Upload your regression dataset (CSV file)", type="csv", key="regression")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(df.head())
        
        # Identify numeric columns for EDA
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        st.write("Numeric Columns:", numeric_cols)
        
        # --- Exploratory Data Analysis (EDA) ---
        with st.expander("Show EDA Visualizations"):
            st.write("### Descriptive Statistics")
            st.write(df.describe())
            
            st.write("### Correlation Matrix")
            corr = df.corr()
            fig_corr, ax_corr = plt.subplots()
            cax = ax_corr.matshow(corr, cmap='viridis')
            fig_corr.colorbar(cax)
            ax_corr.set_xticks(range(len(corr.columns)))
            ax_corr.set_yticks(range(len(corr.columns)))
            ax_corr.set_xticklabels(corr.columns, rotation=90)
            ax_corr.set_yticklabels(corr.columns)
            st.pyplot(fig_corr)
            
            st.write("### Box Plots for Numeric Features")
            for col in numeric_cols:
                fig_box, ax_box = plt.subplots()
                ax_box.boxplot(df[col].dropna())
                ax_box.set_title(f"Box Plot - {col}")
                ax_box.set_ylabel(col)
                st.pyplot(fig_box)
            
            st.write("### Scatter Plots: Features vs. Target")
            target_col_eda = st.selectbox("Select the target column for EDA", options=numeric_cols, key="eda_target")
            if target_col_eda:
                feature_candidates = [col for col in numeric_cols if col != target_col_eda]
                for col in feature_candidates:
                    fig_scatter, ax_scatter = plt.subplots()
                    ax_scatter.scatter(df[col], df[target_col_eda], alpha=0.6)
                    ax_scatter.set_xlabel(col)
                    ax_scatter.set_ylabel(target_col_eda)
                    ax_scatter.set_title(f"{col} vs {target_col_eda}")
                    st.pyplot(fig_scatter)
        
        # --- Data Preprocessing & Modeling ---
        st.subheader("Data Preprocessing and Modeling")
        target_col = st.selectbox("Select the target column for modeling", options=df.columns, key="model_target")
        if target_col:
            # Use only numeric features excluding the target (if numeric)
            feature_cols = [col for col in df.columns if col != target_col and col in numeric_cols]
            st.write("Selected Feature Columns for Modeling:", feature_cols)
            
            # Fill missing values with the mean (for numeric columns)
            X = df[feature_cols].fillna(df[feature_cols].mean())
            if df[target_col].dtype in ['int64', 'float64']:
                y = df[target_col].fillna(df[target_col].mean())
            else:
                st.error("The target column must be numeric for linear regression.")
                st.stop()
            
            st.write("Preprocessed Feature Data Preview:", X.head())
            
            # Train linear regression model
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            
            # Calculate performance metrics
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            st.write(f"Mean Squared Error (MSE): {mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.write(f"R² Score: {r2:.2f}")
            
            # Visualization: Predictions vs. Actual Values with Best Fit Line
            st.subheader("Predictions vs. Actual Values")
            fig_pred, ax_pred = plt.subplots()
            ax_pred.scatter(y, predictions, alpha=0.6, label="Data Points")
            # Best fit line using np.polyfit
            coeffs = np.polyfit(y, predictions, 1)
            x_line = np.linspace(y.min(), y.max(), 100)
            y_line = np.polyval(coeffs, x_line)
            ax_pred.plot(x_line, y_line, color="red", label="Best Fit Line")
            ax_pred.set_xlabel("Actual Values")
            ax_pred.set_ylabel("Predicted Values")
            ax_pred.set_title("Scatter Plot with Best Fit Line")
            ax_pred.legend()
            st.pyplot(fig_pred)
            
            # Additional Visualization: Residual Plot
            st.subheader("Residual Plot")
            residuals = y - predictions
            fig_res, ax_res = plt.subplots()
            ax_res.scatter(predictions, residuals, alpha=0.6)
            ax_res.axhline(y=0, color="red", linestyle="--")
            ax_res.set_xlabel("Predicted Values")
            ax_res.set_ylabel("Residuals")
            ax_res.set_title("Residual Plot")
            st.pyplot(fig_res)
            
            # Custom Prediction Input
            st.subheader("Make a Custom Prediction")
            custom_input = {}
            for col in feature_cols:
                default_val = float(X[col].mean())
                custom_input[col] = st.number_input(f"Enter value for {col}", value=default_val, key=f"input_{col}")
            if st.button("Predict Custom Value"):
                input_vals = [custom_input[col] for col in feature_cols]
                custom_pred = model.predict([input_vals])
                st.write("Predicted Value:", custom_pred[0])
    else:
        st.info("Awaiting CSV file upload for regression analysis.")
        

# ----------------------------
# Clustering Module Function (Enhanced)
# ----------------------------
def clustering_module():
    st.header("Clustering Application")
    
    # File Upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV file) for clustering", type="csv", key="clust_upload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(df.head())
        
        # Identify numeric columns and allow user selection
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        st.write("Numeric Columns:", numeric_cols)
        
        selected_features = st.multiselect("Select features for clustering", options=numeric_cols, default=numeric_cols[:2], key="clust_features")
        if selected_features:
            # Drop missing values based on selected features
            X = df[selected_features].dropna()
            
            st.markdown("### Optional Steps for Increased Accuracy")
            st.markdown("The following options can help improve clustering performance. Hover over each option to learn why it might be beneficial.")
            use_scaling = st.checkbox("Apply Feature Scaling (StandardScaler)", value=True, help="Standardizing features ensures all variables contribute equally to the clustering distance metrics.")
            use_outlier_removal = st.checkbox("Remove Outliers (IQR method)", value=False, help="Removing outliers reduces noise that may distort cluster formation.")
            use_advanced_evaluation = st.checkbox("Perform Advanced Evaluation (Elbow, Silhouette, DB, CH)", value=True, help="Advanced metrics assist in determining the optimal cluster count and overall cluster quality.")
            
            # --- Optionally Remove Outliers ---
            if use_outlier_removal:
                def remove_outliers_IQR(data):
                    Q1 = np.percentile(data, 25, axis=0)
                    Q3 = np.percentile(data, 75, axis=0)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
                    return data[mask]
                X_array = X.values
                X_clean = remove_outliers_IQR(X_array)
                X = pd.DataFrame(X_clean, columns=selected_features)
                st.markdown(f"Outliers have been removed using the IQR method. New data shape: {X.shape}")
            
            # --- Optionally Scale Features ---
            if use_scaling:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.write("Scaled Feature Data Preview:")
                st.write(pd.DataFrame(X_scaled, columns=selected_features).head())
            else:
                X_scaled = X.values
            
            # --- Optional: Advanced Evaluation of Optimal k ---
            if use_advanced_evaluation:
                st.subheader("Optimal Number of Clusters Analysis")
                k_range = st.slider("Select range of k values to analyze", min_value=2, max_value=10, value=(2, 7), key="k_range")
                ks = list(range(k_range[0], k_range[1] + 1))
                inertia_values = []
                silhouette_scores = []
                db_scores = []
                ch_scores = []
                
                for k in ks:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(X_scaled)
                    inertia_values.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(X_scaled, labels))
                    db_scores.append(davies_bouldin_score(X_scaled, labels))
                    ch_scores.append(calinski_harabasz_score(X_scaled, labels))
                
                # Plotting evaluation metrics
                fig_inertia, ax_inertia = plt.subplots()
                ax_inertia.plot(ks, inertia_values, marker="o")
                ax_inertia.set_title("Elbow Method (Inertia)")
                ax_inertia.set_xlabel("Number of clusters (k)")
                ax_inertia.set_ylabel("Inertia")
                st.pyplot(fig_inertia)
                
                fig_sil, ax_sil = plt.subplots()
                ax_sil.plot(ks, silhouette_scores, marker="o")
                ax_sil.set_title("Silhouette Scores")
                ax_sil.set_xlabel("Number of clusters (k)")
                ax_sil.set_ylabel("Silhouette Score")
                st.pyplot(fig_sil)
                
                fig_eval, (ax_db, ax_ch) = plt.subplots(1, 2, figsize=(12, 4))
                ax_db.plot(ks, db_scores, marker="o", color="green")
                ax_db.set_title("Davies-Bouldin Index")
                ax_db.set_xlabel("Number of clusters (k)")
                ax_db.set_ylabel("DB Index (lower is better)")
                ax_ch.plot(ks, ch_scores, marker="o", color="purple")
                ax_ch.set_title("Calinski-Harabasz Score")
                ax_ch.set_xlabel("Number of clusters (k)")
                ax_ch.set_ylabel("CH Score (higher is better)")
                st.pyplot(fig_eval)
            
            # Let the user select the final number of clusters
            k_selected = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3, key="k_selected")
            kmeans = KMeans(n_clusters=k_selected, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Append cluster labels to the original dataset
            df_clusters = df.copy()
            df_clusters["Cluster"] = np.nan
            df_clusters.loc[X.index, "Cluster"] = clusters
            st.subheader("Clustered Data Preview")
            st.write(df_clusters.head())
            
            # --- Visualization ---
            if len(selected_features) == 2:
                st.subheader("2D Scatter Plot of Clusters")
                fig, ax = plt.subplots()
                scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap="viridis", alpha=0.7)
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(selected_features[1])
                ax.set_title("Clusters Visualization")
                plt.colorbar(scatter, label="Cluster")
                st.pyplot(fig)
            elif len(selected_features) >= 3:
                view_option = st.radio("Choose Visualization", ("2D (PCA)", "3D (PCA)"), key="clust_viz")
                if view_option == "2D (PCA)":
                    pca_2d = PCA(n_components=2)
                    X_reduced = pca_2d.fit_transform(X_scaled)
                    st.subheader("2D Scatter Plot (PCA) of Clusters")
                    fig, ax = plt.subplots()
                    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap="viridis", alpha=0.7)
                    ax.set_xlabel("Principal Component 1")
                    ax.set_ylabel("Principal Component 2")
                    ax.set_title("2D Clusters (PCA)")
                    plt.colorbar(scatter, label="Cluster")
                    st.pyplot(fig)
                else:
                    pca_3d = PCA(n_components=3)
                    X_reduced_3d = pca_3d.fit_transform(X_scaled)
                    st.subheader("3D Scatter Plot (PCA) of Clusters")
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    sc = ax.scatter(X_reduced_3d[:, 0], X_reduced_3d[:, 1], X_reduced_3d[:, 2],
                                    c=clusters, cmap="viridis", alpha=0.7)
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_zlabel("PC3")
                    ax.set_title("3D Clusters (PCA)")
                    fig.colorbar(sc, label="Cluster")
                    st.pyplot(fig)
            
            # Download option
            csv_data = df_clusters.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Clustered Dataset",
                data=csv_data,
                file_name='clustered_data.csv',
                mime='text/csv'
            )
    else:
        st.info("Awaiting CSV file upload for clustering analysis.")
        

# ----------------------------
# Neural Network Module Function (Classification)
# ----------------------------
def neural_network_module():
    st.header("Neural Network for Classification")
    st.markdown(
        "Upload a CSV dataset for a classification task. The target column should be categorical. Ensure your features are numeric."
    )
    
    # File upload for classification dataset
    uploaded_file = st.file_uploader("Upload your classification dataset (CSV file)", type="csv", key="nn_upload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(df.head())
        
        # Let user select the target column
        target_column = st.selectbox("Select the target column", options=df.columns, key="nn_target")
        
        # Allow user to choose feature columns (using only numeric columns)
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        available_features = [col for col in numeric_cols if col != target_column]
        selected_features = st.multiselect("Select feature columns", options=available_features, default=available_features, key="nn_features")
        
        if target_column and selected_features:
            # Encode target column as categorical labels
            labels, uniques = pd.factorize(df[target_column])
            df["target_encoded"] = labels
            num_classes = len(uniques)
            st.write("Unique classes found:", uniques)
            
            # Prepare feature data and target
            X = df[selected_features].fillna(df[selected_features].mean())
            y = df["target_encoded"]
            
            # Split data into training and testing sets (80/20 split)
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Standardize features for better neural network training
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Allow user to set hyperparameters
            st.markdown("### Set Hyperparameters")
            epochs = st.number_input("Number of epochs", min_value=1, value=10, step=1, key="nn_epochs")
            learning_rate = st.number_input("Learning Rate", min_value=0.0001, value=0.001, format="%.4f", key="nn_lr")
            batch_size = st.number_input("Batch Size", min_value=1, value=32, step=1, key="nn_batch")
            
            # Build the neural network model using TensorFlow/Keras
            import tensorflow as tf
            model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(X_train_scaled.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            st.subheader("Model Summary")
            # Display the model summary as text
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text("\n".join(model_summary))
            
            # Training the model with a custom callback for real-time progress updates
            st.subheader("Training Progress")
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_text.text(f"Epoch {epoch+1}: Loss = {logs['loss']:.4f}, Accuracy = {logs['accuracy']:.4f}")
                    progress_bar.progress((epoch + 1) / epochs)
            
            with st.spinner("Training Model..."):
                history = model.fit(
                    X_train_scaled, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_test_scaled, y_test),
                    verbose=0,
                    callbacks=[StreamlitCallback()]
                )
            st.success("Training complete!")
            
            # Plot training metrics
            st.markdown("### Training Metrics")
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(history.history['loss'], label="Train Loss")
            ax_loss.plot(history.history['val_loss'], label="Validation Loss")
            ax_loss.set_title("Loss over Epochs")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            st.pyplot(fig_loss)
            
            fig_acc, ax_acc = plt.subplots()
            ax_acc.plot(history.history['accuracy'], label="Train Accuracy")
            ax_acc.plot(history.history['val_accuracy'], label="Validation Accuracy")
            ax_acc.set_title("Accuracy over Epochs")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.legend()
            st.pyplot(fig_acc)
            
            # Custom prediction interface for the trained model
            st.subheader("Make a Custom Prediction")
            custom_inputs = {}
            for feature in selected_features:
                default_val = float(X[feature].mean())
                custom_inputs[feature] = st.number_input(f"Enter value for {feature}", value=default_val, key=f"nn_input_{feature}")
            
            if st.button("Predict Custom Class"):
                input_array = np.array([custom_inputs[feature] for feature in selected_features]).reshape(1, -1)
                input_array_scaled = scaler.transform(input_array)
                prediction_proba = model.predict(input_array_scaled)
                predicted_class = np.argmax(prediction_proba, axis=1)[0]
                st.write("Predicted Class:", uniques[predicted_class])
                st.write("Probability Distribution:", prediction_proba)
    else:
        st.info("Awaiting CSV file upload for neural network classification.")

# ----------------------------
# LLM Q&A Module (Using Gemini API)
# ----------------------------
# Note: Install the gemini package as per Google's Gemini API instructions:
# e.g., pip install google-genai
from google.genai import Client  # Import the Gemini API client
from google import genai
from google.genai import types

# Initialize the client with your API key; replace with your actual API key.
client = Client(api_key="AIzaSyCVIka7igfvXvtFEsMTv3EjmPjScDHJ7W0")  # Make sure to replace YOUR_GEMINI_API_KEY with your Gemini API key

def split_into_chunks(text, chunk_size=500):
    """Split text into chunks, each with approximately chunk_size words."""
    words = text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def retrieve_relevant_chunk(context_text, query, chunk_size=500):
    """Retrieve the chunk from context_text most relevant to query using TF-IDF cosine similarity."""
    chunks = split_into_chunks(context_text, chunk_size)
    # Combine chunks with the query for vectorization
    all_docs = chunks + [query]
    vectorizer = TfidfVectorizer().fit(all_docs)
    chunk_vectors = vectorizer.transform(chunks)
    query_vector = vectorizer.transform([query])
    # Compute cosine similarity between query and each chunk
    similarities = (chunk_vectors * query_vector.T).toarray().flatten()
    best_index = np.argmax(similarities)
    best_chunk = chunks[best_index]
    return best_chunk, similarities[best_index]

def rag_llm_module():
    st.header("LLM Q&A Module with RAG")
    st.markdown("""
    This module uses Google's Gemini API to answer your questions by augmenting your query with context from a selected document.
    Choose a document as context:
      - Academic City Student Handbook
      - 2025 Budget Statement and Economic Policy
    """)

    # Let the user choose the context document (we assume local files here)
    document_choice = st.selectbox("Select Context Document", 
                                     ("Academic City Student Handbook", "2025 Budget Statement and Economic Policy"))
    
    context_text = ""
    context_label = document_choice
    try:
        if document_choice == "Academic City Student Handbook":
            with open("handbook.pdf", "rb") as f:
                context_text = extract_text_from_pdf(f)
        else:
            with open("2025-Budget-Statement-and-Economic-Policy_v4.pdf", "rb") as f:
                context_text = extract_text_from_pdf(f)
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return

    if context_text:
        # Retrieve the most relevant chunk based on the user's later query.
        st.markdown(f"**Using context from:** {context_label}")
    else:
        st.error("No context text loaded.")
        return

    question = st.text_input("Enter your question (answer will be influenced by the selected context):")
    
    # Let the user adjust the maximum tokens if desired
    max_tokens = st.number_input(
        "Max new tokens", 
        min_value=50, 
        max_value=8192,  
        value=150,  
        step=10,
        help="Set the maximum number of tokens to generate for the answer."
    )
    
    if question and st.button("Get Answer"):
        with st.spinner("Retrieving relevant context and generating answer via Gemini API..."):
            try:
                # Retrieve the best matching chunk to the query
                relevant_chunk, sim_score = retrieve_relevant_chunk(context_text, question, chunk_size=500)
                st.markdown("**Retrieved context snippet (most relevant chunk):**")
                st.text(relevant_chunk[:500] + " ...")  # display first 500 characters for preview
                st.markdown(f"*Similarity score: {sim_score:.2f}*")
                
                # Build the prompt combining the retrieved chunk with the user's question
                prompt = f"Context (from {context_label}):\n{relevant_chunk}\n\nQuestion: {question}"
                
                gen_config = types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20
                )
                response = client.models.generate_content(
                    model="gemini-2.0-flash-001",
                    contents=prompt,
                    config=gen_config
                )
                st.markdown("**Answer:**")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error during generation: {e}")



# ----------------------------
# Main Application: Tabs
# ----------------------------
st.title("AI Exam Project")
st.markdown("This unified application includes modules for Regression, Clustering, Neural Networks, and LLM Q&A RAG tasks.")

tabs = st.tabs(["Regression", "Clustering", "Neural Network", "LLM Q&A"])

with tabs[0]:
    regression_module()

with tabs[1]:
    clustering_module()

with tabs[2]:
    neural_network_module()

with tabs[3]:
    rag_llm_module()
