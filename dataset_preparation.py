import requests
import json
import os
from PIL import Image


def create_dataset(input_json_path, output_genres_json_path,
                   output_poster_directory):
    with open(input_json_path, "r") as f:
        movies = json.load(f)

    genres = set()
    labels = {}
    i = 0
    for movie in movies:
        try:
            i += 1
            print(i)
            image_title = movie['movie_title'].strip().lower() + '.jpg'
            temp_genres = movie['genres']
            image_url = movie['image_urls'][0]
            for genre in movie['genres']:
                genres.add(genre)
            im = Image.open(requests.get(image_url, stream=True).raw)
            im.save(output_poster_directory + '/' + image_title)
            labels[image_title] = temp_genres
        except OSError as detail:
            print('Image not found : ' + str(detail))
    ordered_genres = list(genres)
    ordered_genres.sort()
    with open('genres.json', 'w') as outfile:
        json.dump(ordered_genres, outfile)
    with open('labels.json', 'w') as outfile:
        json.dump(labels, outfile)


if __name__ == "__main__":
    create_dataset('imdb_output.json', '', 'assets/posters')
