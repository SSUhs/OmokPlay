# 텐서플로우 지원 테스트

def check_tf(compile_env):
    if compile_env == 'local':
        import tensorflow as tf
        print(f'텐서플로우 버전 : {tf.__version__}')
        devices = tf.config.list_physical_devices()
        print(devices)
    elif compile_env == 'colab':
        import tensorflow as tf
        print(tf.__version__)
        devices = tf.config.list_physical_devices()
        print(devices)
        if devices[0].device_type == 'CPU':  # CPU 환경에서는 학습 X
            print("Colab에서 실행시 GPU 가속 환경으로 변경해주세요")
            quit()
    else:
        print("잘못된 파라미터")
        quit()

