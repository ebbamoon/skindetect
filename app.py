import streamlit as st
import pandas as pd
import openai
import os
from PIL import Image
import numpy as np
import cv2
# from keras.models import load_model
from tensorflow.keras.models import load_model
from ultralytics import YOLO
openai.api_key = os.getenv("OPENAI_API_KEY")

product_df = pd.read_csv('data/product_info.csv')
skintype_df = pd.read_csv('data/skintype.csv')
review_df = pd.read_csv('data/product_review.csv')

## Ch01. 페이지 제목
st.set_page_config(page_title='글로윙(GLOWING)',page_icon="🧴",layout="wide", )
st.write("""
# 글로윙(GLOWING) 얼굴 부위별 피부 진단 AI ✨- 당신의 피부 타입은?
얼굴 사진을 업로드하면 **피부타입 분석, 피부 관리법, 제품 추천**을 제공합니다.
""")

# 모델 로드 함수
@st.cache_resource
def load_models():
    # YOLOv8 모델 로드
    yolo_model = YOLO('str_proj/4_cosmetics/model/best.pt')
    # 각 facepart별 딥러닝 모델 로드
    skin_models = {
        0: {'sensitive': load_model('str_proj/4_cosmetics/model/0/sensitive.h5')},
        1: {
            'forehead_moisture': load_model(
                'str_proj/4_cosmetics/model/1/forehead_moisture_regression.h5'),
            'forehead_elasticity_R2': load_model(
                'str_proj/4_cosmetics/model/1/forehead_elasticity_R2_regression.h5'),
            'forehead_wrinkle': load_model(
                'str_proj/4_cosmetics/model/1/forehead_wrinkle.h5')
        },
        2: {'glabellus_wrinkle': load_model(
            'str_proj/4_cosmetics/model/2/glabellus_wrinkle.h5')},
        3: {'l_perocular_wrinkle': load_model(
            'str_proj/4_cosmetics/model/3/l_perocular_wrinkle.h5')},
        4: {'r_perocular_wrinkle': load_model(
            'str_proj/4_cosmetics/model/4/r_perocular_wrinkle.h5')},
        5: {
            'l_cheek_moisture': load_model(
                'str_proj/4_cosmetics/model/5/l_cheek_moisture_regression.h5'),
            'l_cheek_elasticity_R2': load_model(
                'str_proj/4_cosmetics/model/5/l_cheek_elasticity_R2_regression.h5'),
            'l_cheek_pore': load_model('str_proj/4_cosmetics/model/5/l_cheek_pore_regression.h5')
        },
        6: {
            'r_cheek_moisture': load_model(
                'str_proj/4_cosmetics/model/6/r_cheek_moisture_regression.h5'),
            'r_cheek_elasticity_R2': load_model(
                'str_proj/4_cosmetics/model/6/r_cheek_elasticity_R2_regression.h5'),
            'r_cheek_pore': load_model('str_proj/4_cosmetics/model/6/r_cheek_pore_regression.h5')
        },
        8: {
            'chin_moisture_regression': load_model(
                'str_proj/4_cosmetics/model/7/chin_moisture_regression.h5'),
            'chin_elasticity_R2_regression': load_model(
                'str_proj/4_cosmetics/model/7/chin_elasticity_R2_regression.h5'),
        },
    }
    return yolo_model, skin_models

# 이미지 리사이즈 함수
def resize_image(img, target_size):
    h, w = img.shape[:2]
    ratio = min(target_size / h, target_size / w)
    new_size = (int(w * ratio), int(h * ratio))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    delta_w = target_size - new_size[0]
    delta_h = target_size - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded, (top, left), ratio

# YOLO 예측 함수
def predict_yolo(model, image, conf=0.25, iou=0.45, max_det=10):
    results = model.predict(image, conf=conf, iou=iou, max_det=max_det)
    return results[0]

# 이미지 전처리 함수
def preprocess_image(img):
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# 피부타입 분류 함수
def classify_skin_type(sensitive, hydrated, dry):
    if sensitive == 1 and hydrated == 1 and dry == 0:
        return "OS+ 민지형(민감 지성)"
    elif sensitive == 1 and hydrated == 0 and dry == 0:
        return "OS- 수부민지형(수분 부족형 민감 지성)"
    elif sensitive == 0 and hydrated == 1 and dry == 0:
        return "ON+ 건지형(건강 지성)"
    elif sensitive == 0 and hydrated == 0 and dry == 0:
        return "ON- 수부지형(수분 부족형 지성)"
    elif sensitive == 1 and hydrated == 1 and dry == 1:
        return "DS+ 민건형(민감 건성)"
    elif sensitive == 1 and hydrated == 0 and dry == 1:
        return "DS- 수부민건형(수분 부족형 민감 건성)"
    elif sensitive == 0 and hydrated == 1 and dry == 1:
        return "DN+ 건건형(건강 건성)"
    elif sensitive == 0 and hydrated == 0 and dry == 1:
        return "DN- 수부건형(수분 부족형 건성)"
    else:
        return "Unknown skin type"

# 메인 앱
def main():
    # 모델 로드
    yolo_model, skin_models = load_models()
    # 유분기 선택
    oil_select = st.selectbox('나의 피부 유분기는?', ( '내 피부 유분도를 선택해주세요', '나의 피부의 유분기는 높은 편이다', '나의 피부의 유분기는 낮은 편이다'))
    dry = None
    if oil_select == '나의 피부의 유분기는 높은 편이다':  # Oily
        dry = 0 #False
    elif oil_select == '나의 피부의 유분기는 낮은 편이다':
        dry = 1 #True

    if dry is not None:
        # 파일 업로더
        uploaded_file = st.file_uploader("이미지를 업로드하세요:", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if uploaded_file:
            for file in uploaded_file:
                image = Image.open(file)
                image_np = np.array(image)

            # YOLO 예측을 위한 이미지 리사이즈
            resized_img, (pad_top, pad_left), resize_ratio = resize_image(image_np, 640)
            # YOLO 예측
            results = predict_yolo(yolo_model, resized_img)
            # 결과 시각화
            col1, col2 = st.columns([1,1])
            with col1:
                with st.container():
                    st.subheader(f'피부 부위 인식:')
                    annotated_img = results.plot()
                    st.image(annotated_img, caption="YOLO 검출 결과", use_column_width=True)

            # 각 facepart별 처리
            facepart_results = {}
            chin_image = None
            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id == 8:  # Save the chin image separately
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1 = max(0, int((x1 - pad_left) / resize_ratio))
                    y1 = max(0, int((y1 - pad_top) / resize_ratio))
                    x2 = min(image_np.shape[1], int((x2 - pad_left) / resize_ratio))
                    y2 = min(image_np.shape[0], int((y2 - pad_top) / resize_ratio))
                    if x1 < x2 and y1 < y2:
                        chin_image = image_np[y1:y2, x1:x2]

                if class_id != 7 and class_id not in facepart_results:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1 = max(0, int((x1 - pad_left) / resize_ratio))
                    y1 = max(0, int((y1 - pad_top) / resize_ratio))
                    x2 = min(image_np.shape[1], int((x2 - pad_left) / resize_ratio))
                    y2 = min(image_np.shape[0], int((y2 - pad_top) / resize_ratio))
                    if x1 < x2 and y1 < y2:
                        cropped_img = image_np[y1:y2, x1:x2]
                        if cropped_img.size > 0:
                            facepart_results[class_id] = cropped_img

                # Replace the lips image with chin image if available
                if chin_image is not None:
                    facepart_results[7] = chin_image

            # 피부 진단 결과 출력
            facepart_dict = {0: '얼굴 전체', 1: '이마', 2: '미간', 3: '왼쪽 눈가', 4: '오른쪽 눈가', 5: '왼쪽 볼', 6: '오른쪽 볼', 7: '입술', 8: '턱'}
            feature_translation = {
                'Sensitive': '민감도', 'moisture': '수분',
                'elasticity': '탄력', 'pore': '모공', 'wrinkle': '주름'}
            predictions_dict = {'민감도': [], '수분': [], '탄력': [], '모공': []}
            sensitive = None
            hydrate = None
            with col2:
                with st.container():
                    st.subheader('종합 피부 분석 결과 :')
                    wrinkle_predictions = []
                    for facepart, img in facepart_results.items():
                        if facepart in skin_models:
                            preprocessed_image = preprocess_image(img)
                            for feature, model in skin_models[facepart].items():
                                prediction = model.predict(preprocessed_image)
                                display_name = feature.capitalize()
                                for key, value in feature_translation.items():
                                    if key.lower() in feature.lower():
                                        display_name = value
                                        break
                                if 'wrinkle' in feature:  # 분류
                                    predicted_class = np.argmax(prediction[0])
                                    wrinkle_predictions.append(predicted_class)
                                else:  # 회귀
                                    pred_value = max(0, prediction[0][0])
                                    if display_name in predictions_dict:
                                        predictions_dict[display_name].append(pred_value)

                    if wrinkle_predictions:
                        average_wrinkle = sum(wrinkle_predictions) / len(wrinkle_predictions)
                        st.slider(f'{feature_translation["wrinkle"]} (0: 적음 / 6: 많음)',
                                  0, 6, int(average_wrinkle))
                    for display_name, values in predictions_dict.items():
                        if values:
                            predicted_value = np.mean(values)
                            if display_name == '민감도':
                                st.slider(f'{display_name} (0: 낮음 / 1: 높음)',
                                          1, 0, int(round(predicted_value)))
                                sensitive = 1 if int(round(predicted_value)) == 1 else 0
                            elif display_name == '수분':
                                st.slider(f'{display_name} (0: 부족 / 1: 많음)',
                                          0, 100, int(predicted_value))
                                hydrate = 1 if predicted_value >= 50 else 0
                            elif display_name == '탄력':
                                st.slider(f'{display_name} (0: 부족 / 1: 높음)',
                                          0.0, 1.0, predicted_value, step=0.01)
                            elif display_name == '모공':
                                st.write(f"총 {display_name} 개수: {int(np.sum(values))} 개")
                    skin_type = classify_skin_type(sensitive, hydrate, dry)
                    container = st.container(border=True)
                    container.write(f'**당신의 피부 유형은 ✨{skin_type}✨ 입니다.**')

            # 부위별 사진, 결과 출력
            index = 0
            cols = st.columns(3)
            for facepart, img in facepart_results.items():
                if facepart in skin_models:
                    if index % 3 == 0:
                        if index != 0:
                            st.write('---')
                        cols = st.columns(3)

                    col_index = index % 3
                    col = cols[col_index]
                    col.write(f"**부위별 진단 : [{facepart_dict[facepart]}]**")
                    with col.expander(f"{facepart_dict[facepart]} 사진 보기"):
                        st.image(img, caption=f"{facepart_dict[facepart]}", use_column_width=True)
                    preprocessed_image = preprocess_image(img)
                    for feature, model in skin_models[facepart].items():
                        prediction = model.predict(preprocessed_image)
                        display_name = feature.capitalize()
                        for key, value in feature_translation.items():
                            if key.lower() in feature.lower():
                                display_name = value
                                break

                        unique_key = f'{facepart}_{feature}_{index}'
                        if 'wrinkle' in feature:  # 분류
                            predicted_class = np.argmax(prediction[0])
                            col.slider(f'{display_name} (0: 적음 / 6: 많음)',
                                       0, 6, int(predicted_class),
                                        key=unique_key)
                        else:  # 회귀
                            predicted_value = max(0, prediction[0][0])
                            if display_name == '민감도':
                                col.slider(f'{display_name} (0: 낮음 / 1: 높음)', 1, 0, int(round(predicted_value)), key=unique_key)
                            elif display_name == '수분':
                                col.slider(f'{display_name} (0: 부족 / 100: 많음)', 0, 100, int(predicted_value), key=unique_key)
                            elif display_name == '탄력':
                                col.slider(f'{display_name} (0: 부족 / 1: 높음)', 0.0, 1.0, float(predicted_value), step=0.01, key=unique_key)
                            elif display_name == '모공':
                                col.write(f"{display_name} 개수: {int(predicted_value)}개")
                    index += 1

            # 고객 리뷰 기반 제품 추천
            if skin_type is not None:
                skintypes = list(skintype_df['name'])
                default_index = skintypes.index(skin_type) if skin_type in skintypes else 0
                selected = st.selectbox(f'피부 타입 선택', skintypes, index=default_index)
                if selected:
                    selected_row = skintype_df[skintype_df['name'] == selected].iloc[0]
                    cat1 = '#Dry' if selected_row['Dry'] == 1 else '#Oily'
                    cat2 = '#Hydrated' if selected_row['Hydrated'] == 1 else '#Dehydrated'
                    cat3 = '#Sensitive' if selected_row['Sensitivity'] == 1 else '#Non-Sensitive'
                    selected_categories = f"{cat1} {cat2} {cat3}"
                    st.subheader(selected_categories)
                    text = skintype_df[skintype_df['name'] == selected]['descript'].iloc[0]
                    st.markdown(f"""
                         <div style="border:1px solid #dcdcdc; padding: 10px; border-radius: 5px;">
                             {text}
                         </div>
                         """, unsafe_allow_html=True)
                    st.divider()

                    top_eight = [review_df.iloc[index]['product_name'] for index in
                                 review_df[str(selected_row['skintype'])].sort_values(ascending=False)[:8].index]
                    features_set = set(list(product_df['product_name']))
                    plus_set = set(top_eight)
                    overlap = list(features_set.intersection(plus_set))
                    filtered_df = product_df[product_df['product_name'].isin(overlap)]

                    # Ch03. 제품 추천 섹션
                    tab_names = ['고객 리뷰 기반 추천', '제품 정보 기반 추천']
                    tabs = st.tabs(tab_names)
                    def display_products_or_reviews(tab_type):
                        if tab_type == '고객 리뷰':
                            st.subheader(f"나와 같은 피부 타입의 사람들이 선호한 제품은?")
                        else:
                            st.subheader(f"내 피부 타입의 사람들을 위해 만들어진 제품은?")
                        st.write(f'**{tab_type} 데이터**의 **키워드 유사도**를 바탕으로 스킨케어 상위 8개 제품을 추천해드려요😊')
                        for i in range(0, len(filtered_df), 4):
                            cols = st.columns(4)
                            for j in range(4):
                                if i + j < len(filtered_df):
                                    with cols[j]:
                                        row = filtered_df.iloc[i + j]
                                        st.write(f"**:blue[#{row['category']}]**")
                                        st.markdown(f"""
                                            <a href="{row['page_url']}" target="_blank">
                                                <img src="{row['image_url']}" alt="{row['product_name']}" style="width:100%">
                                            </a>
                                            """, unsafe_allow_html=True)
                                        st.write('   ')
                                        st.write(f"**제품명: {row['product_name']}**")
                                        st.write(f"**브랜드:** {row['brand']}")
                                        st.write(f"**가격:** {row['price']}")

                            # After displaying a row of 4 products, write a title for the next set
                            if i + 4 < len(filtered_df):
                                st.write('---')
                                # st.subheader(f":blue[#에센스/세럼/앰플] 제품 추천 - {tab_type} 유사도")

                        st.write('   ')
                        st.write('   ')
                        st.write('   ')

                    with tabs[0]:
                        display_products_or_reviews('고객 리뷰')
                    with tabs[1]:
                        display_products_or_reviews('제품 정보')

if __name__ == "__main__":
    main()