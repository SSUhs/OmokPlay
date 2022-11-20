# 학습 데이터 위치 모아주기
import os


ends_ext = ['.rec','.psq','.REC','.PSQ']


if __name__ == '__main__':
    dir_path = input("폴더 절대 경로 : ")
    for (root, directories, files) in os.walk(dir_path):
        for d in directories:
            d_path = os.path.join(root, d)
            print(d_path)

        for file in files:
            file_path = os.path.join(root, file)
            if file_path.endswith('.zip'):
                print(f'압축이 안풀린 파일 : ',file_path);
                quit()
            print(file_path)