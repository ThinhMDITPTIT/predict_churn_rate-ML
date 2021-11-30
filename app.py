import streamlit as stlit
import pandas as pdas
import joblib
model = joblib.load(r"./notebook/model.sav")
from preprocessing_data import preprocess_data

def main():
    # Tiêu đề ứng dụng
    stlit.title("Ứng dụng dự đoán tỉ lệ Churn")

    # Thanh sidebar
    sb_selectbox = stlit.sidebar.selectbox(
        "Hình thức dự đoán?", ("Online", "Batch")
    )

    if sb_selectbox == "Online":
        stlit.subheader("Thông tin nhân khẩu học khách hàng:")
        seniorcitizen = stlit.selectbox("Người cao tuổi:", ("Yes", "No"))
        dependents = stlit.selectbox("Có người phụ thuộc:", ("Yes", "No"))

        stlit.subheader("Các dịch vụ khách hàng đăng kí:")
        mutliplelines = stlit.selectbox(
            "Nhiều đường truyền:", ("Yes", "No", "No phone service")
        )
        phoneservice = stlit.selectbox("Dịch vụ di động:", ("Yes", "No"))
        internetservice = stlit.selectbox(
            "Dịch vụ Internet:", ("DSL", "Fiber optic", "No")
        )
        onlinesecurity = stlit.selectbox(
            "Dịch vụ bảo mật trực tuyến:",
            ("Yes", "No", "No internet service"),
        )
        onlinebackup = stlit.selectbox(
            "Dịch vụ sao lưu trực tuyến:", ("Yes", "No", "No internet service")
        )
        techsupport = stlit.selectbox(
            "Dịch vụ hỗ trợ công nghệ:",
            ("Yes", "No", "No internet service"),
        )
        streamingtv = stlit.selectbox(
            "Dịch vụ truyền hình trực tuyến:", ("Yes", "No", "No internet service")
        )
        streamingmovies = stlit.selectbox(
            "Dịch vụ xem phim trực tuyến:", ("Yes", "No", "No internet service")
        )

        stlit.subheader("Thông tin tài khoản khách hàng:")
        tenure = stlit.slider(
            "Thời gian khách hàng đã gắn bó với doanh nghiệp (đơn vị: tháng):",
            min_value=0,
            max_value=72,
            value=0,
        )
        contract = stlit.selectbox(
            "Loại hợp đồng:", ("Month-to-month", "One year", "Two year")
        )
        paperlessbilling = stlit.selectbox("Thanh toán không hoá đơn:", ("Yes", "No"))
        PaymentMethod = stlit.selectbox(
            "Phương thức thanh toán:",
            (
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ),
        )
        monthlycharges = stlit.number_input(
            "Chi phí chi trả hàng tháng:",
            min_value=0,
            max_value=150,
            value=0,
        )
        totalcharges = stlit.number_input(
            "Tổng chi phí chi trả:",
            min_value=0,
            max_value=10000,
            value=0,
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
        features_dtframe = pdas.DataFrame.from_dict([data])
        stlit.markdown("<h3></h3>", unsafe_allow_html=True)
        stlit.write("Tổng quan dữ liệu đầu vào:")
        stlit.markdown("<h3></h3>", unsafe_allow_html=True)
        stlit.dataframe(features_dtframe)

        # Preprocess inputs
        preprocess_dtframe = preprocess_data(features_dtframe, "Online")
        prediction = model.predict(preprocess_dtframe)

        if stlit.button("Dự đoán"):
            if prediction == 1:
                stlit.warning("Có, khách hàng sẽ ngưng sử dụng dịch vụ.")
            else:
                stlit.success("Không, khách hàng hài lòng với dịch vụ.")

    else:
        stlit.subheader("Dataset upload")
        uploaded_file = stlit.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pdas.read_csv(uploaded_file)
            uploaded_file.seek(0)
            stlit.write(data.head())
            stlit.markdown("<h3></h3>", unsafe_allow_html=True)
            preprocess_dtframe = preprocess_data(data, "Batch")
            if stlit.button("Dự đoán"):
                prediction = model.predict(preprocess_dtframe)
                prediction_df = pdas.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace(
                    {
                        1: "Có, khách hàng sẽ ngưng sử dụng dịch vụ.",
                        0: "Không, khách hàng hài lòng với dịch vụ.",
                    }
                )

                stlit.markdown("<h3></h3>", unsafe_allow_html=True)
                stlit.subheader("Dự đoán")
                stlit.write(prediction_df)

if __name__ == "__main__":
    main()