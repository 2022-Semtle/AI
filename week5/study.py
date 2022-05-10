# NUMPY
# - 파이썬의 써드파티 모듈
# - 복잡한 행렬계산, 선형대수, 통계등의 기능 제공
# - 고속 연산 수행
# - 더 자세하게 공부하려면 아래 링크 참조
# - https://teddylee777.github.io/python/numpy-tutorial

# 설치
# - 콘솔창에 아래 명령어 입력
# - pip install numpy

# 모듈 임포트
# - 일반적으로 alias는 np를 사용
import numpy as np

print(f'module : {np.__name__}')
print(f'version: {np.__version__}')


# 배열 생성
some_list = [1, 2, 3]
# some_list = [[1, 2, 3], [4, 5, 6]]  # 여러 가지 리스트를 사용해보세요.
array = np.array(some_list)
print(f'array : {array}')

# 배열 속성
print(f'array.shape: {array.shape}')  # 배열의 모양, 튜플 형태로 반환
print(f'array.dtype: {array.dtype}')  # 배열 원소의 타입
                                      # 파이썬은 기본적으로 타입을 자동 추정하지만, 속도가 느립니다.
                                      # NUMPY는 타입을 강제하는 대신 속도를 선택합니다.
print(f'array.size : {array.size}')   # 배열의 크기, 총 원소의 개수

# 유용한 기능
print(f'transpose: {array.T}')


# random, slicing, fancy indexing
# broadcasting, matrix operator
# where, axis 같은 기능에 대해서 공부하면
# 단순히 numpy 뿐 아니라 코딩에 대한 
# 기본적인 이해가 많이 높아집니다!

# 저번에 했던 활성화 함수(activation function)을
# 넘파이 배열을 입력으로 받도록 수정합니다.
# 기존 함수는 단일값을 받았던 것을 기억하세요!
def step_function(x):
    y = x > 0
    return y.astype(np.int64)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)


# >>> 여기서 부터 
# 두 개의 원소 x1, x2가 있습니다.
# 두 원소는 입력이 됩니다.
# 또한 두 개의 입력을 받는 3개의 퍼셉트론이 있습니다.

class perceptron:
    def __init__(self, w1, w2, bias, activation=True):
        self.w1 = w1
        self.w2 = w2
        self.bias = bias
        self.activation = activation
        
    def __call__(self, x1, x2):
        y = x1*self.w1 + x2*self.w2 + self.bias
        if self.activation:
            # step function
            if y > 0:
                return 1
            else:
                return 0
        else:
            return y

x1 = -0.7
x2 = 0.3
perceptron1 = perceptron(-0.1, 0.2, 0.3)
perceptron2 = perceptron(0.3, -0.1, -0.1)
perceptron3 = perceptron(0.2, 0.7, -0.3)

# 각 퍼셉트론의 출력은 다음과 같은 계산될 수 있습니다.

a1 = perceptron1(x1, x2)
a2 = perceptron2(x1, x2)
a3 = perceptron2(x1, x2)
print('일반 연산:', a1, a2, a3)

# 어렵지 않죠?
# 아직 이게 왜 필요한 과정이지? 왜 이게 딥러닝을 공부하는데
# 필요하지? 라는 의문이 있을 수 있습니다만
# 조금만 인내하고 따라와 주세요!

# 각 입력은 여러분이 생각하는 바에 따라 다양해질 수 있습니다.
# 키와 몸무게, 위도와 경도 등...
# 출력도 정의하기 나름입니다
# 그러면 입력이 이미지라고 생각하면 어떻게 될까요?
# 요즘 핸드폰 카메라 해상도가 높아서 FHD(1920*1080) 해상도는 평범합니다.
# FHD 해상도를 표현하려면 8bit 값이 1920(넓이) * 1080(높이) * 3(RGB)개가 필요하고
# 이는 x가 1920 * 1080 * 3 = 6220800개 필요함을 의미합니다.

# 위와 같은 구조로는 구현도 어렵고, 유지보수도 힘들며
# 무엇보다 느립니다.
# 따라서 넘파이를 이용한 행렬 곱으로 다음과 같이
# 표현할 수 있습니다.

# <<< 여기까지, 넘파이 관련 없이 작성됩니다.

# >>> 여기서부터

# X 입력, W 가중치, B 편향(bias)
X = np.array([-0.7, 0.3])
W = np.array([[-0.1, 0.3, 0.2], [0.2, -0.1, 0.7]])
B = np.array([0.3, -0.1, -0.3])

# @은 행렬곱 연산자로 사용됩니다.
# X.dot(W)으로 구현해도 동일합니다.
# 위에서 작성한 step_function 함수도 활용합니다.
Y = step_function(X@W + B)
print('넘파이 연산:', Y)

# <<< 여기까지, 위 구조를 넘파이로 작성합니다.


# 결과가 똑같이 나오죠?
# 이런 X, W, B 로 이루어진 것을 하나의 층(layer)라고 부릅니다.
# challenge 4를 확인해보고 도전해보세요!