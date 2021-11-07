import json

from ngram_utils.prepare_profiles import prepare_profiles

langs = ['eng', 'rus']
output_dir = 'output/ngram'

profiles = prepare_profiles(langs)

for lang, profile in profiles.items():
    file = open(f'{output_dir}/{lang}.json', 'w')
    json.dump(profile, file)
    file.close()
