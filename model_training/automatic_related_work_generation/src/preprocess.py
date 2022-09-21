import config
from prepro import data_builder

if __name__ == "__main__":
	eval(f'data_builder.format_to_{config.target_model}()')