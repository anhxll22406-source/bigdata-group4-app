import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import os
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns

# --- CẤU HÌNH CHUNG ---
st.set_page_config(page_title="Big Data App: Phân Tích & Dự Đoán Churn", layout="wide")

# Load model
@st.cache_resource
def load_my_model():
    model_path = "best_model.pkl"
    if not os.path.exists(model_path):
        st.error("Không tìm thấy model! Hãy đảm bảo file 'best_model.pkl' ở cùng thư mục.")
        return None
    return joblib.load(model_path)

model = load_my_model()

# Kết nối MongoDB
@st.cache_resource
def connect_mongo():
    uri = "mongodb+srv://anhxll22406_db_user:n0WSOLVB8EpYFpmS@bigdata-group4.2masnqr.mongodb.net/?appName=bigdata-group4"
    client = MongoClient(uri)
    db = client['dataset-bigdata']
    collection = db['group4-bigdata']
    return collection

collection = connect_mongo()

# Fetch data từ MongoDB dùng Pandas
@st.cache_data
def load_data_from_mongo():
    projection = {
        "_id": 0,
        "CustomerID": 1,
        "Age": 1,
        "Gender": 1,
        "Tenure": 1,
        "Usage Frequency": 1,
        "Support Calls": 1,
        "Payment Delay": 1,
        "Subscription Type": 1,
        "Contract Length": 1,
        "Total Spend": 1,
        "Last Interaction": 1,
        "Churn": 1
    }
    cursor = collection.find({}, projection)
    df = pd.DataFrame(list(cursor))
    if df.empty:
        st.warning("Không có dữ liệu từ MongoDB. Kiểm tra kết nối hoặc collection.")
    return df

# --- TRANG CHỦ ---
st.markdown("<h1 style='text-align: center; color: #0D47A1;'>Ứng Dụng Big Data: Phân Tích & Dự Đoán Churn Khách Hàng</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    if st.button("Phân Tích Dữ Liệu (từ MongoDB)", use_container_width=True):
        st.session_state.mode = "Analyze"
        st.rerun()
with col2:
    if st.button("Dự Đoán Churn (Mô Hình ML)", use_container_width=True):
        st.session_state.mode = "Predict"
        st.rerun()

if "mode" not in st.session_state:
    st.session_state.mode = None

# --- PHẦN 1: PHÂN TÍCH DỮ LIỆU (DÙNG PANDAS) ---
if st.session_state.mode == "Analyze":
    st.subheader("Phân Tích Dữ Liệu Khách Hàng Churn")
    df = load_data_from_mongo()

    if df.empty:
        st.error("Không có dữ liệu từ MongoDB. Kiểm tra kết nối hoặc collection.")
    else:
        # Thống kê cơ bản
        st.write("**Thống Kê Cơ Bản:**")
        stats = {
            'Tổng khách hàng': len(df),
            'Tỷ lệ churn (%)': (df[df['Churn'] == 1].shape[0] / len(df)) * 100 if len(df) > 0 else 0,
            'Tuổi trung bình churn': df[df['Churn'] == 1]['Age'].mean() if not df[df['Churn'] == 1].empty else 0,
            'Tuổi trung bình không churn': df[df['Churn'] == 0]['Age'].mean() if not df[df['Churn'] == 0].empty else 0,
            'Chi tiêu trung bình churn': df[df['Churn'] == 1]['Total Spend'].mean() if not df[df['Churn'] == 1].empty else 0,
            'Chi tiêu trung bình không churn': df[df['Churn'] == 0]['Total Spend'].mean() if not df[df['Churn'] == 0].empty else 0,
            'Số Support Calls trung bình churn': df[df['Churn'] == 1]['Support Calls'].mean() if not df[df['Churn'] == 1].empty else 0,
        }
        st.write(stats)

        # Bar chart churn theo Subscription Type
        st.write("**Tỷ Lệ Churn Theo Subscription Type:**")
        churn_rate_sub = df.groupby('Subscription Type')['Churn'].mean().reset_index()
        fig_bar, ax_bar = plt.subplots()
        sns.barplot(data=churn_rate_sub, x='Subscription Type', y='Churn', ax=ax_bar, palette='viridis')
        ax_bar.set_title('Tỷ Lệ Churn Theo Loại Subscription')
        ax_bar.set_ylabel('Tỷ lệ churn')
        st.pyplot(fig_bar)

        # Histogram Age theo Churn
        st.write("**Phân Bố Tuổi Theo Churn:**")
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(data=df, x='Age', hue='Churn', multiple='stack', kde=True, ax=ax_hist, palette='viridis')
        ax_hist.set_title('Phân Bố Tuổi Theo Churn')
        st.pyplot(fig_hist)

        # Scatter plot Age vs Total Spend, màu theo Churn
        st.write("**Age vs Total Spend (màu theo Churn):**")
        fig_scatter, ax_scatter = plt.subplots()
        sns.scatterplot(data=df, x='Age', y='Total Spend', hue='Churn', palette='viridis', alpha=0.7, ax=ax_scatter)
        ax_scatter.set_title('Age vs Total Spend')
        st.pyplot(fig_scatter)

        # Filter interactive
        st.header("Filter Interactive")
        min_age = st.slider("Lọc Age lớn hơn:", 18, 100, 30)
        filtered_df = df[df['Age'] > min_age]
        st.write(f"Dữ liệu sau filter (Age > {min_age}) - Top 10:")
        st.dataframe(filtered_df.head(10))

    if st.button("Quay Lại Trang Chủ"):
        st.session_state.mode = None
        st.rerun()

# --- PHẦN 2: DỰ ĐOÁN CHURN ---
elif st.session_state.mode == "Predict":
    # Khởi tạo session state cho predict
    if "predict_mode" not in st.session_state:
        st.session_state.predict_mode = "Single"
    if "step" not in st.session_state:
        st.session_state.step = 1
    if "bulk_step" not in st.session_state:
        st.session_state.bulk_step = "Upload"
    if "show_raw_data" not in st.session_state:
        st.session_state.show_raw_data = False

    default_values = {
        "Age": 35, "Gender": "Male", "Tenure": 12, "Usage Frequency": 15,
        "Support Calls": 2, "Payment Delay": 5, "Last Interaction": 10,
        "Subscription Type": "Standard", "Contract Length": "Annual", "Total Spend": 500.0
    }
    if "form_data" not in st.session_state:
        st.session_state.form_data = default_values.copy()

    def sync_slider_to_input(feature):
        st.session_state.form_data[feature] = st.session_state[f"sl_{feature}"]

    def sync_input_to_slider(feature):
        st.session_state.form_data[feature] = st.session_state[f"in_{feature}"]

    def reset_predict():
        st.session_state.step = 1
        st.session_state.form_data = default_values.copy()
        st.session_state.predict_mode = "Single"
        st.session_state.bulk_step = "Upload"
        st.session_state.show_raw_data = False

    st.subheader("Dự Đoán Churn Khách Hàng")

    st.session_state.predict_mode = st.radio("Chọn chế độ dự đoán:", ("Single (1 khách hàng)", "Bulk (Hàng loạt từ file)"))

    if st.session_state.predict_mode == "Single (1 khách hàng)":
        progress_map = {1: 25, 2: 50, 3: 75, 4: 100}
        st.progress(progress_map[st.session_state.step] / 100)

        if st.session_state.step == 1:
            st.subheader("Bước 1: Thông tin cá nhân")
            st.session_state.form_data["Age"] = st.slider("Age (tuổi)", 18, 100, st.session_state.form_data["Age"])
            st.session_state.form_data["Gender"] = st.selectbox("Gender", ["Male", "Female"], 
                                                                index=0 if st.session_state.form_data["Gender"] == "Male" else 1)
            if st.button("Tiếp theo", use_container_width=True):
                st.session_state.step = 2
                st.rerun()

        elif st.session_state.step == 2:
            st.subheader("Bước 2: Hành vi sử dụng")
            def render_sync_row(feature, label, min_v, max_v):
                col_slider, col_input = st.columns([3, 1])
                with col_slider:
                    st.slider(label, min_v, max_v, key=f"sl_{feature}", 
                              value=st.session_state.form_data[feature],
                              on_change=sync_slider_to_input, args=(feature,))
                with col_input:
                    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
                    st.number_input(label, min_v, max_v, key=f"in_{feature}", 
                                    value=st.session_state.form_data[feature],
                                    label_visibility="collapsed",
                                    on_change=sync_input_to_slider, args=(feature,))

            render_sync_row("Tenure", "Tenure (tháng sử dụng)", 0, 72)
            render_sync_row("Usage Frequency", "Usage Frequency (lần/tháng)", 0, 50)
            render_sync_row("Support Calls", "Support Calls (số lần gọi)", 0, 20)
            render_sync_row("Payment Delay", "Payment Delay (ngày chậm trả)", 0, 30)
            render_sync_row("Last Interaction", "Last Interaction (ngày tương tác cuối)", 0, 30)

            c1, c2 = st.columns(2)
            with c1: 
                if st.button("Quay lại", use_container_width=True): st.session_state.step = 1; st.rerun()
            with c2: 
                if st.button("Tiếp theo", use_container_width=True): st.session_state.step = 3; st.rerun()

        elif st.session_state.step == 3:
            st.subheader("Bước 3: Đăng ký & Chi tiêu")
            subs = ["Basic", "Standard", "Premium"]
            contracts = ["Monthly", "Quarterly", "Annual"]
            st.session_state.form_data["Subscription Type"] = st.selectbox("Subscription Type", subs, 
                                    index=subs.index(st.session_state.form_data["Subscription Type"]))
            st.session_state.form_data["Contract Length"] = st.selectbox("Contract Length", contracts,
                                    index=contracts.index(st.session_state.form_data["Contract Length"]))
            st.session_state.form_data["Total Spend"] = st.number_input("Total Spend ($)", min_value=0.0, value=float(st.session_state.form_data["Total Spend"]))

            c1, c2 = st.columns(2)
            with c1: 
                if st.button("Quay lại", use_container_width=True): st.session_state.step = 2; st.rerun()
            with c2: 
                if st.button("Dự đoán ngay", use_container_width=True): st.session_state.step = 4; st.rerun()

        elif st.session_state.step == 4:
            st.subheader("Thông tin khách hàng")
            df_summary = pd.DataFrame([st.session_state.form_data]).T
            df_summary.columns = ["Giá trị"]
            st.table(df_summary)

            if model:
                input_df = pd.DataFrame([st.session_state.form_data])
                features = ["Age", "Gender", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Subscription Type", "Contract Length", "Total Spend", "Last Interaction"]
                input_df = input_df[features]
                
                prob = model.predict_proba(input_df)[0][1]
                prediction = model.predict(input_df)[0]
                
                color = "#4CAF50" if prob < 0.3 else ("#FFC107" if prob < 0.7 else "#F44336")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob*100, number={'suffix': "%"},
                    title={'text': "Xác suất rời bỏ"},
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}}
                ))
                st.plotly_chart(fig, use_container_width=True)

                res = "DỰ ĐOÁN KHÁCH HÀNG SẼ RỜI BỎ" if prediction == 1 else "DỰ ĐOÁN KHÁCH HÀNG SẼ Ở LẠI"
                risk = "Low Risk 🟢" if prob < 0.3 else ("Medium Risk 🟡" if prob < 0.7 else "High Risk 🔴")
                st.markdown(f"<h2 style='text-align: center; color: {color};'>{res}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center;'>Mức độ rủi ro: <b>{risk}</b></p>", unsafe_allow_html=True)
            else:
                st.error("Không tìm thấy model!")

            if st.button("Thực hiện dự đoán mới", use_container_width=True): reset_predict(); st.rerun()

    else:  # Bulk mode
        if st.session_state.bulk_step == "Upload":
            st.subheader("Dự báo hàng loạt từ File")
            uploaded_file = st.file_uploader("Tải file dữ liệu (CSV hoặc Excel)", type=["csv", "xlsx"])

            if uploaded_file:
                st.success("Tải file lên thành công!")
                if uploaded_file.name.endswith('.csv'):
                    df_input = pd.read_csv(uploaded_file)
                else:
                    df_input = pd.read_excel(uploaded_file)
                
                st.session_state.data_to_predict = df_input

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Chuyển sang dự đoán single", use_container_width=True):
                        st.session_state.predict_mode = "Single (1 khách hàng)"
                        st.session_state.step = 1
                        st.rerun()
                with col2:
                    btn_label = "Ẩn dữ liệu" if st.session_state.show_raw_data else "Hiển thị dữ liệu import"
                    if st.button(btn_label, use_container_width=True):
                        st.session_state.show_raw_data = not st.session_state.show_raw_data
                        st.rerun()
                with col3:
                    if st.button("Dự đoán ngay", type="primary", use_container_width=True):
                        st.session_state.bulk_step = "Result"
                        st.rerun()
                
                if st.session_state.show_raw_data:
                    st.markdown("### Dữ liệu đã import:")
                    st.dataframe(df_input, use_container_width=True)

        elif st.session_state.bulk_step == "Result":
            st.subheader("Kết quả dự đoán danh sách khách hàng")
            df = st.session_state.data_to_predict

            if model:
                try:
                    features = ["Age", "Gender", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Subscription Type", "Contract Length", "Total Spend", "Last Interaction"]
                    probs = model.predict_proba(df[features])[:, 1]
                    
                    result_df = pd.DataFrame()
                    result_df["Customer ID"] = df.get("CustomerID", df.index + 1)
                    result_df["Tỷ lệ rời bỏ (%)"] = [f"{p*100:.2f}%" for p in probs]
                    result_df["Mức độ rủi ro"] = ["🟢 Low Risk" if p < 0.3 else ("🟡 Medium Risk" if p < 0.7 else "🔴 High Risk") for p in probs]                
                    st.table(result_df)
                    
                except Exception as e:
                    st.error(f"Lỗi: File không đúng định dạng các cột cần thiết. Chi tiết: {e}")
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Chuyển sang dự đoán single", use_container_width=True):
                    reset_predict()
                    st.rerun()
            with c2:
                if st.button("Dự đoán hàng loạt mới", use_container_width=True):
                    st.session_state.bulk_step = "Upload"
                    st.session_state.show_raw_data = False
                    st.rerun()

    if st.button("Quay Lại Trang Chủ"):
        st.session_state.mode = None
        reset_predict()
        st.rerun()