from util import process_image_folder
from util import write_master_list
from util import generator
import unittest


#TODO
#class MyTest(unittest.TestCase):
#    def test_process_image_folder(self):
#        lst = process_image_folder('test/IMG/')
#        self.assertEqual(len(lst), 3)

def main():
    lst = process_image_folder('test/')
    print(len(lst))
    write_master_list('test/ml.csv', lst)



    train_generator = generator(lst, 2)
    for x, y in train_generator:
        print(x.shape)

#def cleanup():
    #remove fliped images
    #clean ml.csv

if __name__ == '__main__':
    main()
