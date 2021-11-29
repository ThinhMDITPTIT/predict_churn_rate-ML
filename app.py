import streamlit as stlit
import pandas as pdas

# Load the predict churn model
import joblib

model = joblib.load(r"./notebook/model.sav")

# Import preprocessing scripts
from preprocessing import preprocess


def main():
    # Setting Application title
    stlit.title("Customer Churn Prediction App")

    # Application description
    """
    This Streamlit app is made to predict customer churn in a ficitional telecommunication use case.
    The application is functional for both online prediction and batch data prediction.
    """

    # Setting Application sidebar default
    add_selectbox = stlit.sidebar.selectbox(
        "How would you like to predict?", ("Online", "Batch")
    )

    if add_selectbox == "Online":
        # Based on our optimal features selection
        stlit.subheader("Demographic data")
        seniorcitizen = stlit.selectbox("Senior Citizen:", ("Yes", "No"))
        dependents = stlit.selectbox("Dependent:", ("Yes", "No"))

        stlit.subheader("Payment data")
        tenure = stlit.slider(
            "Number of months the customer has stayed with the company",
            min_value=0,
            max_value=72,
            value=0,
        )
        contract = stlit.selectbox(
            "Contract", ("Month-to-month", "One year", "Two year")
        )
        paperlessbilling = stlit.selectbox("Paperless Billing", ("Yes", "No"))
        PaymentMethod = stlit.selectbox(
            "PaymentMethod",
            (
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ),
        )
        monthlycharges = stlit.number_input(
            "The amount charged to the customer monthly",
            min_value=0,
            max_value=150,
            value=0,
        )
        totalcharges = stlit.number_input(
            "The total amount charged to the customer",
            min_value=0,
            max_value=10000,
            value=0,
        )

        stlit.subheader("Services signed up for")
        mutliplelines = stlit.selectbox(
            "Does the customer have multiple lines", ("Yes", "No", "No phone service")
        )
        phoneservice = stlit.selectbox("Phone Service:", ("Yes", "No"))
        internetservice = stlit.selectbox(
            "Does the customer have internet service", ("DSL", "Fiber optic", "No")
        )
        onlinesecurity = stlit.selectbox(
            "Does the customer have online security",
            ("Yes", "No", "No internet service"),
        )
        onlinebackup = stlit.selectbox(
            "Does the customer have online backup", ("Yes", "No", "No internet service")
        )
        techsupport = stlit.selectbox(
            "Does the customer have technology support",
            ("Yes", "No", "No internet service"),
        )
        streamingtv = stlit.selectbox(
            "Does the customer stream TV", ("Yes", "No", "No internet service")
        )
        streamingmovies = stlit.selectbox(
            "Does the customer stream movies", ("Yes", "No", "No internet service")
        )

        data = {
            "SeniorCitizen": seniorcitizen,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phoneservice,
            "MultipleLines": mutliplelines,
            "InternetService": internetservice,
            "OnlineSecurity": onlinesecurity,
            "OnlineBackup": onlinebackup,
            "TechSupport": techsupport,
            "StreamingTV": streamingtv,
            "StreamingMovies": streamingmovies,
            "Contract": contract,
            "PaperlessBilling": paperlessbilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": monthlycharges,
            "TotalCharges": totalcharges,
        }
        features_df = pdas.DataFrame.from_dict([data])
        stlit.markdown("<h3></h3>", unsafe_allow_html=True)
        stlit.write("Overview of input is shown below")
        stlit.markdown("<h3></h3>", unsafe_allow_html=True)
        stlit.dataframe(features_df)

        # Preprocess inputs
        preprocess_df = preprocess(features_df, "Online")

        prediction = model.predict(preprocess_df)

        if stlit.button("Predict"):
            stlit.write(prediction)
            if prediction == 1:
                stlit.warning("Yes, the customer will terminate the service.")
            else:
                stlit.success("No, the customer is happy with services.")

    else:
        stlit.subheader("Dataset upload")
        uploaded_file = stlit.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pdas.read_csv(uploaded_file)
            uploaded_file.seek(0)
            # Get overview of data
            stlit.write(data.head())
            stlit.markdown("<h3></h3>", unsafe_allow_html=True)
            # Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if stlit.button("Predict"):
                # Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pdas.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace(
                    {
                        1: "Yes, the customer will terminate the service.",
                        0: "No, the customer is happy with services.",
                    }
                )

                stlit.markdown("<h3></h3>", unsafe_allow_html=True)
                stlit.subheader("Prediction")
                stlit.write(prediction_df)


if __name__ == "__main__":
    main()