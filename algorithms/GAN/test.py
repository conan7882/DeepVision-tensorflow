import config

def testfnc(con):
	print(con.data_dir)


if __name__ == '__main__':
	testfnc(config)
	print(type(config))