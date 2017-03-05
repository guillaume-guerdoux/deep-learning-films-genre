import os


if __name__ == "__main__":
    absolute_path = 'assets/posters/'
    for folder in os.listdir(absolute_path):
        for filename in os.listdir(absolute_path + folder):
            if folder + '.jpg' != filename:
                os.remove(absolute_path + folder + '/' + filename)
        break
