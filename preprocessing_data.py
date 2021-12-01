from sklearn.preprocessing import MinMaxScaler
import pandas as pdas

# Tiền xử lý dữ liệu đầu vào:
def preprocess_data(dtframe, churn_opt):
    def binary_map(feature):
        return feature.map({"Yes": 1, "No": 0})

    # Chuẩn hoá các thuộc tính
    binary_list = ["SeniorCitizen", "Dependents", "PhoneService", "PaperlessBilling"]
    dtframe[binary_list] = dtframe[binary_list].apply(binary_map)

    # Chuẩn hoá dữ liệu sử dụng MinMaxScaler
    scale = MinMaxScaler()

    if churn_opt == "Online":
        columns = [
            "SeniorCitizen",
            "Dependents",
            "tenure",
            "PhoneService",
            "PaperlessBilling",
            "MonthlyCharges",
            "TotalCharges",
            "MultipleLines_No_phone_service",
            "MultipleLines_Yes",
            "InternetService_Fiber_optic",
            "InternetService_No",
            "OnlineSecurity_No_internet_service",
            "OnlineSecurity_Yes",
            "OnlineBackup_No_internet_service",
            "TechSupport_No_internet_service",
            "TechSupport_Yes",
            "StreamingTV_No_internet_service",
            "StreamingTV_Yes",
            "StreamingMovies_No_internet_service",
            "StreamingMovies_Yes",
            "Contract_One_year",
            "Contract_Two_year",
            "PaymentMethod_Electronic_check",
        ]
        # Chuẩn hoá các trường thuộc tính có nhiều hơn 2 loại giá trị
        dtframe = pdas.get_dummies(dtframe).reindex(columns=columns, fill_value=0)
        
        temp_min = {'tenure': 0, 'MonthlyCharges': 0, 'TotalCharges': 0}
        temp_max = {'tenure': 72, 'MonthlyCharges': 150, 'TotalCharges': 10000}
        dtframe = dtframe.append(temp_min, ignore_index = True)
        dtframe = dtframe.append(temp_max, ignore_index = True)
        dtframe["tenure"] = scale.fit_transform(dtframe[["tenure"]])
        dtframe["MonthlyCharges"] = scale.fit_transform(dtframe[["MonthlyCharges"]])
        dtframe["TotalCharges"] = scale.fit_transform(dtframe[["TotalCharges"]])
        dtframe = dtframe.iloc[:-2 , :]
    elif churn_opt == "Batch":
        pass
        dtframe = dtframe[
            [
                "SeniorCitizen",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "MonthlyCharges",
                "TotalCharges",
            ]
        ]
        columns = [
            "SeniorCitizen",
            "Dependents",
            "tenure",
            "PhoneService",
            "PaperlessBilling",
            "MonthlyCharges",
            "TotalCharges",
            "MultipleLines_No_phone_service",
            "MultipleLines_Yes",
            "InternetService_Fiber_optic",
            "InternetService_No",
            "OnlineSecurity_No_internet_service",
            "OnlineSecurity_Yes",
            "OnlineBackup_No_internet_service",
            "TechSupport_No_internet_service",
            "TechSupport_Yes",
            "StreamingTV_No_internet_service",
            "StreamingTV_Yes",
            "StreamingMovies_No_internet_service",
            "StreamingMovies_Yes",
            "Contract_One_year",
            "Contract_Two_year",
            "PaymentMethod_Electronic_check",
        ]
        # Chuẩn hoá các trường thuộc tính có nhiều hơn 2 loại giá trị
        dtframe = pdas.get_dummies(dtframe).reindex(columns=columns, fill_value=0)

        dtframe["tenure"] = scale.fit_transform(dtframe[["tenure"]])
        dtframe["MonthlyCharges"] = scale.fit_transform(dtframe[["MonthlyCharges"]])
        dtframe["TotalCharges"] = scale.fit_transform(dtframe[["TotalCharges"]])
    else:
        print("Lựa chọn không tồn tại.")

    return dtframe