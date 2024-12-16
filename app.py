# Streamlit 라이브러리 임포트
import streamlit as st

# 기본 구조
# st.title("Hello, Streamlit!")
# st.write("This is a simple Streamlit app.")
# st.write("Let's add some more text !")

# # 메인 페이지 내용
# st.title("Streamlit 레이아웃 설정 예제")
# st.write("이 예제는 Streamlit의 다양한 레이아웃 기능을 보여줍니다.")

# # 사이드바 사용 예시
# st.sidebar.title("사이드바")
# st.sidebar.write("여기에 사이드바 내용을 추가할 수 있습니다.")

# # 사이드바에 슬라이더 추가
# sidebar_value = st.sidebar.slider("사이드바 슬라이더", 0, 100, 25)
# st.sidebar.write(f"선택한 값: {sidebar_value}")

# # 컬럼 사용 예시
# cols = st.columns(2)

# with cols[0]:
#     st.header("컬럼 1")
#     st.write("여기는 첫 번째 컬럼입니다.")
#     st.button("컬럼 1 버튼")

# with cols[1]:
#     st.header("컬럼 2")
#     st.write("여기는 두 번째 컬럼입니다.")
#     st.button("컬럼 2 버튼")

# # 확장기 사용 예시
# with st.expander("더보기"):
#     st.write("여기에 추가 정보를 넣을 수 있습니다.")
#     st.image("https://via.placeholder.com/150", caption="예시 이미지")


# # 제목과 설명 추가
# st.title("Streamlit 주요 개념 및 구조 이해")
# st.write("이 앱은 Streamlit의 주요 개념과 구조를 설명합니다.")

# # 위젯 사용 예시
# if st.button("클릭하세요"):
#     st.write("버튼이 클릭되었습니다!")

# # 슬라이더 사용 예시
# value = st.slider("값을 선택하세요", 0, 100, 50)
# st.write(f"선택한 값: {value}")

# # 데이터 시각화 예시
# import pandas as pd
# import numpy as np

# data = pd.DataFrame(
#     np.random.randn(value, 3),
#     columns=['a', 'b', 'c']
# )

# st.line_chart(data)

# import streamlit as st
# import plotly.express as px
# import pandas as pd

# # 샘플 데이터 생성
# df = pd.DataFrame({
#     "x": [1, 2, 3, 4, 5],
#     "y": [10, 11, 12, 13, 14],
#     "category": ["A", "B", "A", "B", "A"]
# })

# # Plotly를 사용하여 산점도 생성
# fig = px.scatter(df, x="x", y="y", color="category", title="Plotly와 Streamlit 연동 예제")

# # Streamlit 앱에 Plotly 그래프 표시
# st.title("Plotly 연동 예제")
# st.plotly_chart(fig)

# st.markdown("# 큰 제목")
# st.markdown("**굵은 텍스트**")
# st.markdown("*기울임 텍스트*")

# st.markdown('<p style="color:blue; font-size:20px;">파란색 텍스트</p>', unsafe_allow_html=True)

# st.title("마크다운 문법 예제")
# st.write("이 앱은 다양한 마크다운 문법을 살펴보는 예제입니다.")

# st.write('리스트')

# st.markdown("""
# - 항목 1
# - 항목 2
#     - 하위 항목 2.1
# 1. 첫 번째
# 2. 두 번째
# """)

# st.write('테이블')

# st.markdown("""
# | 헤더 1 | 헤더 2 |
# |--------|--------|
# | 값 1   | 값 2   |
# | 값 3   | 값 4   |
# """)


# st.write('링크')

# st.markdown("[Streamlit 공식 사이트](https://streamlit.io)")

# st.write('이미지')

# st.markdown("![대체 텍스트](https://via.placeholder.com/150)")

# import streamlit as st
# import pandas as pd

# # 샘플 데이터프레임 생성
# df = pd.DataFrame({
#     "이름": ["홍길동", "이순신", "강감찬"],
#     "점수": [90, 85, 88]
# })

# st.table(df)

# st.metric(label="온도", value="25°C", delta="1.2°C")
# st.metric(label="습도", value="60%", delta="-5%")

# import pandas as pd

# # 샘플 데이터프레임 생성
# data = {
#     "이름": ["홍길동", "이순신", "강감찬", "유관순", "장보고"],
#     "점수": [90, 85, 88, 92, 78],
#     "과목": ["수학", "과학", "영어", "역사", "수학"]
# }
# df = pd.DataFrame(data)

# # Streamlit 앱 제목
# # st.title("데이터프레임 필터링 및 정렬 예제")

# # # 과목별 필터링
# # subject_filter = st.selectbox("과목 선택", options=df["과목"].unique())
# # filtered_df = df[df["과목"] == subject_filter]

# # # 점수 정렬
# # sort_order = st.radio("정렬 순서 선택", options=["오름차순", "내림차순"])
# # ascending = True if sort_order == "오름차순" else False
# # sorted_df = filtered_df.sort_values(by="점수", ascending=ascending)

# # 필터링 및 정렬된 데이터프레임 출력
# st.dataframe(df)


# sample_json = {
#     "이름": "홍길동",
#     "나이": 30,
#     "주소": {
#         "도시": "서울",
#         "구": "강남구"
#     }
# }

# st.json(sample_json)

# import pandas as pd

# # CSV 파일 불러오기
# @st.cache_data
# def load_data():
#     return pd.read_csv('cars.csv')

# # 데이터 로드
# df = load_data()

# # 제목과 설명 추가
# st.title("자동차 데이터")
# st.markdown('<p style="font-weight:bold; color:green;">자동차 데이터 테이블</p>', unsafe_allow_html=True)

# # 필터링: 제조사 선택
# manufacturer = st.selectbox("제조사 선택", options=df['Manufacturer'].unique())
# filtered_df = df[df['Manufacturer'] == manufacturer]

# # 정렬: 컬럼 선택
# sort_column = st.selectbox("정렬할 컬럼 선택", options=df.columns)
# sort_order = st.radio("정렬 순서 선택", options=["오름차순", "내림차순"])
# ascending = True if sort_order == "오름차순" else False
# sorted_df = filtered_df.sort_values(by=sort_column, ascending=ascending)

# # 데이터프레임 출력
# st.dataframe(sorted_df)


############################################################ 시각화

# import matplotlib.pyplot as plt
# import numpy as np

# # 데이터 생성
# x = np.linspace(0, 10, 100)
# y = np.sin(x)

# # Matplotlib 그래프 생성
# fig, ax = plt.subplots()
# ax.plot(x, y, label='Sine Wave')
# ax.set_title('Matplotlib Example')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.legend()

# # Streamlit에 그래프 표시
# st.pyplot(fig)


# import streamlit as st
# import seaborn as sns
# import pandas as pd

# # 샘플 데이터 로드
# df = sns.load_dataset('iris')

# # Seaborn 그래프 생성
# fig = sns.pairplot(df, hue='species')

# # Streamlit에 그래프 표시
# st.pyplot(fig)

# import streamlit as st
# import plotly.express as px
# import pandas as pd

# # 샘플 데이터 생성
# data = {
#     "Category": ["A", "B", "C", "D"],
#     "Values": [10, 20, 15, 25]
# }
# df = pd.DataFrame(data)

# # Plotly를 사용하여 막대 그래프 생성
# fig = px.bar(df, x="Category", y="Values", title="Plotly Bar Chart Example")

# # Streamlit에 그래프 표시
# st.plotly_chart(fig)

# # 설명 추가
# st.write("이 그래프는 각 카테고리의 값을 보여주는 간단한 막대 그래프입니다.")

# import streamlit as st
# import plotly.express as px
# import pandas as pd

# # 샘플 데이터 생성
# data = {
#     "Category": ["A", "B", "C", "D"],
#     "Values": [10, 20, 15, 25]
# }
# df = pd.DataFrame(data)

# # Streamlit 슬라이더를 사용하여 값 조정
# value_a = st.slider("Category A", 0, 50, 10)
# value_b = st.slider("Category B", 0, 50, 20)
# value_c = st.slider("Category C", 0, 50, 15)
# value_d = st.slider("Category D", 0, 50, 25)

# # 슬라이더로 조정된 값을 데이터프레임에 반영
# df["Values"] = [value_a, value_b, value_c, value_d]

# # Plotly를 사용하여 막대 그래프 생성
# # fig = px.bar(df, x="Category", y="Values", title="Interactive Plotly Bar Chart")

# fig = sns.barplot(df, x="Category", y="Values")

# # Streamlit에 그래프 표시
# st.pyplot(fig)

# # 설명 추가
# st.write("슬라이더를 사용하여 각 카테고리의 값을 조정할 수 있습니다.")

# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np

# # Streamlit 슬라이더를 사용하여 파라미터 조정
# frequency = st.slider("Frequency", 1, 10, 5)
# amplitude = st.slider("Amplitude", 1, 10, 1)

# # 데이터 생성
# x = np.linspace(0, 10, 100)
# y = amplitude * np.sin(frequency * x)

# # Matplotlib 그래프 생성
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title(f'Sine Wave: Frequency={frequency}, Amplitude={amplitude}')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')

# # Streamlit에 그래프 표시
# st.pyplot(fig)

# # 설명 추가
# st.write("슬라이더를 사용하여 사인파의 주파수와 진폭을 조정할 수 있습니다.")

# import streamlit as st
# import plotly.express as px
# import pandas as pd

# # 샘플 데이터 생성
# df = pd.DataFrame({
#     "x": [1, 2, 3, 4, 5],
#     "y": [10, 11, 12, 13, 14],
#     "z": [5, 4, 3, 2, 1],
#     "category": ["A", "B", "A", "B", "A"]
# })

# # 3D 산점도 생성
# fig = px.scatter_3d(df, x='x', y='y', z='z', color='category', title='3D Scatter Plot')

# # Streamlit에 그래프 표시
# st.plotly_chart(fig)


# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import numpy as np

# # 샘플 데이터 생성
# np.random.seed(42)
# n_points = 100
# df = pd.DataFrame({
#     "x": np.random.rand(n_points),
#     "y": np.random.rand(n_points),
#     "frame": np.repeat(np.arange(1, 11), n_points // 10),
#     "category": np.random.choice(["A", "B", "C"], n_points)
# })

# # 애니메이션이 있는 산점도 생성
# fig = px.scatter(
#     df, 
#     x='x', 
#     y='y', 
#     color='category', 
#     animation_frame='frame', 
#     animation_group='category',
#     title='Animated Scatter Plot with Longer Data'
# )

# # Streamlit에 그래프 표시
# st.plotly_chart(fig)

# # 설명 추가
# st.write("이 그래프는 시간에 따라 변화하는 데이터를 보여줍니다. 각 프레임은 다른 시점을 나타냅니다.")

# import streamlit as st
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 샘플 데이터 로드
# df = sns.load_dataset('tips')

# # FacetGrid 생성
# g = sns.FacetGrid(df, col='time', row='sex', margin_titles=True)
# g.map(sns.scatterplot, 'total_bill', 'tip')

# # Streamlit에 그래프 표시
# st.pyplot(g)

# import streamlit as st
# import matplotlib.pyplot as plt
# import numpy as np

# # Streamlit 슬라이더를 사용하여 파라미터 조정
# frequency = st.slider("Frequency", 1, 10, 5)
# amplitude = st.slider("Amplitude", 1, 10, 1)

# # 데이터 생성
# x = np.linspace(0, 10, 100)
# y = amplitude * np.sin(frequency * x)

# # Matplotlib 그래프 생성
# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.set_title(f'Sine Wave: Frequency={frequency}, Amplitude={amplitude}')
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')

# # Streamlit에 그래프 표시
# st.pyplot(fig)

# # 설명 추가
# st.write("슬라이더를 사용하여 사인파의 주파수와 진폭을 조정할 수 있습니다.")


# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import numpy as np

# # 샘플 데이터 생성
# np.random.seed(42)
# n_points = 100
# df = pd.DataFrame({
#     "x": np.random.rand(n_points),
#     "y": np.random.rand(n_points),
#     "category": np.random.choice(["A", "B", "C"], n_points)
# })

# # Streamlit 선택 상자를 사용하여 카테고리 선택
# selected_category = st.selectbox("카테고리 선택", options=df["category"].unique())

# # 선택한 카테고리에 해당하는 데이터 필터링
# filtered_df = df[df["category"] == selected_category]

# # Plotly 산점도 생성
# fig = px.scatter(filtered_df, x='x', y='y', color='category', title=f'Scatter Plot for Category {selected_category}')

# # Streamlit에 그래프 표시
# st.plotly_chart(fig)

# # 설명 추가
# st.write("선택 상자를 사용하여 다른 카테고리의 데이터를 볼 수 있습니다.")

# 샘플 데이터 생성
# np.random.seed(42)
# n_points = 100
# df = pd.DataFrame({
#     "x": np.random.rand(n_points),
#     "y": np.random.rand(n_points),
#     "category": np.random.choice(["A", "B", "C"], n_points)
# })

# # Seaborn 색상 팔레트 목록
# color_palettes = ["deep", "muted", "pastel", "dark", "colorblind"]

# # Streamlit 선택 상자를 사용하여 색상 팔레트 선택
# selected_palette = st.selectbox("Seaborn 색상 팔레트 선택", options=color_palettes)

# # 선택한 팔레트에 따라 색상 설정
# palette_colors = sns.color_palette(selected_palette, n_colors=3).as_hex()

# # Plotly 산점도 생성
# fig = px.scatter(df, x='x', y='y', color='category', title='Scatter Plot with Seaborn Color Palette',
#                  color_discrete_sequence=palette_colors)

# # Streamlit에 그래프 표시
# st.plotly_chart(fig)

# # 설명 추가
# st.write("선택 상자를 사용하여 Seaborn의 색상 팔레트를 적용할 수 있습니다.")

# import streamlit as st
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Iris 데이터셋 로드
# iris = sns.load_dataset('iris')

# # Matplotlib 산점도 생성
# fig, ax = plt.subplots()
# ax.scatter(iris['sepal_length'], iris['sepal_width'], c='blue', label='Sepal')
# ax.set_xlabel('Sepal Length')
# ax.set_ylabel('Sepal Width')
# ax.set_title('Iris Sepal Dimensions')
# ax.legend()

# # Streamlit에 그래프 표시
# st.pyplot(fig)

# # Seaborn 히스토그램 생성
# fig, ax = plt.subplots()
# sns.histplot(iris['petal_length'], bins=20, kde=True, ax=ax)
# ax.set_title('Petal Length Distribution')

# # Streamlit에 그래프 표시
# st.pyplot(fig)

# # Seaborn 박스플롯 생성
# fig, ax = plt.subplots()
# sns.boxplot(x='species', y='petal_length', data=iris, ax=ax)
# ax.set_title('Petal Length by Species')

# # Streamlit에 그래프 표시
# st.pyplot(fig)

# import plotly.express as px

# # Plotly 산점도 생성
# fig = px.scatter(iris, x='sepal_length', y='sepal_width', color='species', title='Interactive Iris Sepal Scatter Plot')
# st.plotly_chart(fig)

# # Plotly 라인 차트 생성
# fig = px.line(iris, x='sepal_length', y='sepal_width', color='species', title='Interactive Iris Sepal Line Chart')
# st.plotly_chart(fig)


# # Plotly 슬라이더와 드롭다운 메뉴를 사용하여 데이터 동적 변경
# fig = px.scatter(iris, x='sepal_length', y='sepal_width', color='species', title='Interactive Iris Sepal Scatter Plot with Slider')

# # 슬라이더 추가
# fig.update_layout(
#     sliders=[{
#         'steps': [{'label': str(i), 'method': 'update', 'args': [{'visible': [True if j == i else False for j in range(len(iris['species'].unique()))]}]} for i in range(len(iris['species'].unique()))],
#         'currentvalue': {'prefix': 'Species: '}
#     }]
# )

# # 드롭다운 메뉴 추가
# fig.update_layout(
#     updatemenus=[{
#         'buttons': [
#             {'label': 'All', 'method': 'update', 'args': [{'visible': [True] * len(iris['species'].unique())}]
#         }
#         ] +
#         [
#             {'label': species, 'method': 'update', 'args': [{'visible': [species == s for s in iris['species']]}]} for species in iris['species'].unique()
#         ],
#         'direction': 'down'
#     }]
# )

# # Streamlit에 그래프 표시
# # st.plotly_chart(fig)

# import streamlit as st
# import plotly.express as px
# import seaborn as sns

# # Iris 데이터셋 로드
# iris = sns.load_dataset('iris')

# # Plotly 산점도 생성
# fig = px.scatter(iris, x='sepal_length', y='sepal_width', color='species', title='Interactive Iris Sepal Scatter Plot with Dropdown')

# # 드롭다운 메뉴 추가
# fig.update_layout(
#     updatemenus=[
#         {
#             'buttons': [
#                 {'label': 'All', 'method': 'update', 'args': [{'visible': [True, True, True]}, {'title': 'All Species'}]},
#                 {'label': 'Setosa', 'method': 'update', 'args': [{'visible': [True, False, False]}, {'title': 'Setosa'}]},
#                 {'label': 'Versicolor', 'method': 'update', 'args': [{'visible': [False, True, False]}, {'title': 'Versicolor'}]},
#                 {'label': 'Virginica', 'method': 'update', 'args': [{'visible': [False, False, True]}, {'title': 'Virginica'}]}
#             ],
#             'direction': 'down',
#             'showactive': True,
#         }
#     ]
# )

# # Streamlit에 그래프 표시
# st.plotly_chart(fig)


######################################################### 상태 관리 및 캐싱

import streamlit as st

# if 'counter' not in st.session_state:
#     st.session_state.counter = 0

# if st.button('Increment'):
#     st.session_state.counter += 1

# st.write(f"Counter: {st.session_state.counter}")

# ### rerun 적용을 위해 나중에 작성
# st.write('상태 관리를 적용했습니다.')

# import streamlit as st
# import pandas as pd
# import time

# start_time = time.time()

# # 주석 처리
# @st.cache_data
# def load_data():
#     return pd.read_csv('2019-Nov.csv')

# data = load_data()
# st.write(data.head())

# end_time = time.time()

# # 캐싱 전, 캐싱 후 소요 시간 비교
# st.write(f'load 소요 시간 = {end_time - start_time}')

# # 캐싱 후 위젯 추가
# st.write('캐시 적용한 뒤 값 넣기')

# import streamlit as st

# name = st.text_input("Enter your name")
# age = st.number_input("Enter your age", min_value=0, max_value=99)

# st.write(f"Hello, {name}. You are {age} years old.")



# import streamlit as st

# show_chart = st.checkbox("Show chart")
# if show_chart:
#     st.line_chart([1, 2, 3, 4, 5])


# import streamlit as st
# from sqlalchemy import create_engine
# import pandas as pd

# # SQLite 데이터베이스 연결
# engine = create_engine('sqlite:///mydatabase.db')

# # 데이터 조회
# @st.cache_data
# def load_data():
#     query = "SELECT * FROM my_table"
#     return pd.read_sql(query, engine)

# data = load_data()
# st.write(data.head())

# import streamlit as st
# from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
# import pandas as pd
# from faker import Faker

# # SQLite 데이터베이스 연결
# engine = create_engine('sqlite:///users.db')
# metadata = MetaData()

# # 테이블 정의
# users_table = Table('users', metadata,
#                     Column('id', Integer, primary_key=True),
#                     Column('name', String),
#                     Column('email', String),
#                     Column('address', String))

# # 테이블 생성
# metadata.create_all(engine)

# # Faker를 사용하여 가짜 데이터 생성
# fake = Faker()

# def generate_fake_data(n=10):
#     with engine.connect() as conn:
#         # 기존 데이터 삭제
#         conn.execute(users_table.delete())
#         # 가짜 데이터 삽입
#         for _ in range(n):
#             conn.execute(users_table.insert().values(
#                 name=fake.name(),
#                 email=fake.email(),
#                 address=fake.address()
#             ))
#         conn.commit()

# # 가짜 데이터 생성 버튼
# if st.button('Generate Fake Data'):
#     generate_fake_data(20)
#     st.success('Fake data generated!')

# # 데이터 조회
# @st.cache_data
# def load_data():
#     with engine.connect() as conn:
#         query = "SELECT * FROM users"
#         return pd.read_sql(query, conn)
        
# # 데이터 로드 및 표시
# data = load_data()
# st.write(data)


# import streamlit as st
# from sqlalchemy import create_engine, text
# import pandas as pd
# from streamlit_autorefresh import st_autorefresh

# # SQLite 데이터베이스 연결
# engine = create_engine('sqlite:///users.db')

# def remove_all_data():
#     with engine.connect() as conn:
#         conn.execute(text("DELETE FROM users"))
#         conn.commit()

# # 데이터 삭제 버튼
# if st.button('Remove All Data'):
#     remove_all_data()
#     st.error('All data removed!')

# # 데이터 조회
# def load_data():
#     with engine.connect() as conn:
#         query = "SELECT * FROM users"
#         return pd.read_sql(query, conn)

# # 주기적으로 데이터 갱신
# st_autorefresh(interval=5000)  # 5초마다 갱신

# # 데이터 로드 및 표시
# data = load_data()
# st.write(data)



st.title("My First Streamlit App")

st.write("Hello Streamlit !")
