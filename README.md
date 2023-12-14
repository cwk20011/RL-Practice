# DDPG (Deep Deterministic Policy Gradient)로 로봇 팔 각도 제어하기

## 목적
이 프로젝트는 DDPG 알고리즘을 사용하여 로봇 팔의 각도를 제어하는 강화 학습 모델을 구현하고 훈련하는 것을 목표로 합니다. 로봇 팔은 초기 각도에서 시작하여 최종 각도에 도달하는데, 에이전트는 주어진 상태에서 최적의 액션을 선택하여 팔을 움직입니다.

## 구조
프로젝트는 다음과 같은 주요 구성 요소로 구성되어 있습니다.

- **`RobotArmModel` 클래스:** 초기화된 각도를 가진 로봇 팔을 모델링합니다.

- **`ActorNet` 클래스:** state를 받아 액션을 출력하는 Actor 네트워크를 정의합니다.

- **`CriticNet` 클래스:** state와 action을 받아 Q값을 출력하는 Critic 네트워크를 정의합니다.

- **`Memory` 클래스:** ReplayBuffer를 뜻하며, Transition을 저장합니다.

- **`Transition` 클래스:** 에이전트의 경험을 저장하는 데이터 구조를 뜻합니다.

- **`DDPGAgent` 클래스:** Actor 및 Critic 네트워크를 초기화하고, 학습 및 탐험을 수행하는 DDPG 에이전트를 정의합니다.

- **`Main` 클래스:** 전체 학습 프로세스를 실행합니다. 에피소드를 반복하면서 에이전트를 훈련하고 결과를 시각화합니다.

## 주요 하이퍼파라미터

- `noise_std`: 액션에 추가되는 탐험용 노이즈의 표준 편차.

- `target_update_freq`: 타겟 네트워크를 업데이트하는 주기.

- `memory_capacity`: ReplayBuffer 용량.

## 에이전트 환경 및 상호작용

### 환경
로봇 팔 모델링이 주어진 환경으로, 초기 각도는 0도부터 360도 사이에서 랜덤으로 설정됩니다.

### 액션
에이전트는 현재 상태를 기반으로 액션을 선택하며, 선택된 액션은 로봇 팔의 각도를 변경합니다.

### 상태
로봇 팔의 현재 각도로 정의되며, 이 값은 환경의 현재 상태를 나타냅니다.

### 리워드
로봇 팔의 최종 각도에 따라 계산되는 리워드가 존재합니다. 추가로, 종료 조건이 충족되면 보상이 더해집니다. 최종 각도에 도달하거나 종료 조건을 만족하면 에피소드가 종료됩니다.

## 리워드 설명

로봇 팔의 각도에 따른 리워드는 다음과 같이 계산됩니다.

1. **기본 리워드:** 현재 각도에서 180도를 뺀 값의 음의 절댓값으로 계산됩니다. 즉, 로봇 팔이 180도에 가까워질수록 리워드가 높아집니다.

2. **종료 조건 리워드:** 최종 각도가 180도에 가까워지면 추가적인 보상이 주어집니다. 이는 종료 조건을 만족하는 경우 해당 보상이 더해집니다.

따라서, 에이전트는 로봇 팔을 180도로 정렬하는 방향으로 학습됩니다.
