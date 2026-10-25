import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="산업 설비 고장 예측",
    page_icon="⚙️"
)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    lr = joblib.load(BASE_DIR / "model_lr.pkl")
    rf = joblib.load(BASE_DIR / "model_rf.pkl")
    return lr, rf

st.title("⚙️ 산업 설비 고장 예측 시스템")
st.write("설비 정보를 입력하고 고장 여부를 예측해 보세요.")

st.info("교육 및 탐구용 시제품이며, 실제 산업 안전 판단에 사용하지 마세요.")

try:
    model_lr, model_rf = load_models()
except Exception as e:
    st.error(f"모델 파일 로딩 오류: {e}")
    st.stop()

machine_type = st.selectbox("설비 유형", ["L", "M", "H"])
air_temp = st.number_input("공기 온도 (K)", value=300.0)
process_temp = st.number_input("공정 온도 (K)", value=310.0)
speed = st.number_input("회전 속도 (rpm)", value=1500)
torque = st.number_input("토크 (Nm)", value=40.0)
wear = st.number_input("공구 마모 시간 (min)", value=100)

model_name = st.selectbox(
    "예측 모델", ["랜덤 포레스트", "로지스틱 회귀"]
)

threshold = st.slider(
    "고장 판정 임계값", 0.05, 0.95, 0.30, 0.05
)

if st.button("고장 예측하기"):
    X = pd.DataFrame([{
        "Type_H": int(machine_type == "H"),
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M"),
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear
    }])

    model = model_rf if model_name == "랜덤 포레스트" else model_lr

    X = X.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    failure_index = list(model.classes_).index(1)
    score = model.predict_proba(X)[0][failure_index]

    st.metric("모델의 고장 점수", f"{score * 100:.1f}%")

    if score >= threshold:
        st.error("고장으로 예측했습니다.")
    else:
        st.success("정상으로 예측했습니다.")

    st.caption(
        "고장 점수는 실제 고장 확률이나 안전을 보장하지 않습니다."
    )
