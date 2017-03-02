import requests
import json
import os
from PIL import Image


def create_dataset(input_json_path, output_genres_json_path,
                   output_poster_directory):
    with open(input_json_path, "r") as f:
        movies = json.load(f)

    genres = set()
    dataset = {}
    i = 0
    for movie in movies:
        try:
            i += 1
            print(i)
            temp_title = movie['movie_title'].strip().lower()
            temp_genres = movie['genres']
            image_url = movie['image_urls'][0]
            for genre in movie['genres']:
                genres.add(genre)
            im = Image.open(requests.get(image_url, stream=True).raw)
            os.mkdir(output_poster_directory + '/' + temp_title)
            im.save(output_poster_directory + '/' + temp_title + '/' + temp_title + '.jpg')
            dataset[temp_title] = temp_genres
        except OSError:
            print('Image not found')
    ordered_genres = list(genres)
    ordered_genres.sort()
    with open('assets/genres.json', 'w') as outfile:
        json.dump(ordered_genres, outfile)
    with open('assets/dataset.json', 'w') as outfile:
        json.dump(dataset, outfile)


if __name__ == "__main__":
    create_dataset('assets/imdb_output.json', '', 'assets/posters')
