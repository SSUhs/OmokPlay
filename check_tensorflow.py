# 텐서플로우 지원 테스트

def check_tf(compile_env):
    if compile_env == 'local':
        import tensorflow as tf
        print(f'텐서플로우 버전 : {tf.__version__}')
        devices = tf.config.list_physical_devices()
        print(devices)
    elif compile_env == 'colab':
        import tensorflow as tf
        print("\n\n---------------------------------------------------")
        print(f'텐서 플로우 버전 : {tf.__version__}\n')
        devices_all = tf.config.list_physical_devices()
        print(f'활성 Device : {devices_all}\n')
        device_name = tf.test.gpu_device_name()

        # GPU를 사용하거나 TPU를 사용하지 않으면 종료 (혹시나 가속기를 안켜놓았을 상황을 방지)
        if device_name != '/device:GPU:0': # GPU 안쓰는 환경인 경우
            mode = int(input("현재 GPU 가속이 미사용 상태입니다. CPU 환경으로만 진행하려면 0, 종료할려면 1"))
            if mode != 0:
                print("종료 합니다")
                quit()
        else:
            print('Found GPU at: {}'.format(device_name))
        # TPU 사용하게 되면 TPU도 예외 처리
        # https://colab.research.google.com/notebooks/tpu.ipynb#scrollTo=FpvUOuC3j27n
        # else:
        #     try:
        #         tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        #         print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
        #     except ValueError:
        #         raise BaseException(
        #             'TPU 연결 오류 발생\nERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
        #     tf.config.experimental_connect_to_cluster(tpu)
        #     tf.tpu.experimental.initialize_tpu_system(tpu)
        #     tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        print("잘못된 파라미터")
        quit()

