import random

def shuffle_data(in_file, out_file):
	outf = open(out_file, 'w')
	with open(in_file) as f:
		lines = f.readlines()
		random.shuffle(lines)
		for line in lines:
			outf.write(line)
	outf.close()

if __name__ == '__main__':
	shuffle_data(in_file = 'sample_64_20_desc.txt', out_file = 'shuffle_sample_64_20_desc.txt')
	shuffle_data(in_file = 'sample_32_10_desc.txt', out_file = 'shuffle_sample_32_10_desc.txt')