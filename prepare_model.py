import json

import seaborn as sns
from matplotlib import pyplot as plt

from neural_utils.prepare_model import prepare_model

langs = ['eng', 'rus']
output_dir = 'output/neural'
model_name = 'en_ru_model'

vocabulary, model, test_info = prepare_model(langs)

model.save(f'{output_dir}/{model_name}', overwrite=True)

vocab_file = open(f'{output_dir}/vocabulary.json', 'w')
json.dump(vocabulary, vocab_file)
vocab_file.close()

print(f'Model accuracy: {test_info["accuracy"]}')
plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
sns.set(font_scale=1.5)
sns.heatmap(test_info['conf_matrix_df'], cmap='coolwarm', annot=True, fmt='.5g', cbar=False)
plt.xlabel('Predicted', fontsize=22)
plt.ylabel('Actual', fontsize=22)
plt.show()
