# 텐서플로우 지원 테스트

def check_tf(compile_env):
    if compile_env == 'local':
        import tensorflow as tf
        print(f'텐서플로우 버전 : {tf.__version__}')
        devices = tf.config.list_physical_devices()
        print(devices)
    elif compile_env == 'colab':
        import tensorflow as tf
        print("\n")
        print(tf.__version__)
        devices = tf.config.list_physical_devices()
        print(devices)
        print(f'디바이스 : {devices[0]}')
    else:
        print("잘못된 파라미터")
        quit()

