import numpy as np
import ForHome1.numpy.datasets as datasets


def main():
    baseball = [180, 215, 210, 210, 188, 176, 209, 200]
    baseball_arr = np.array(baseball)
    print(f"Baseball array: {baseball_arr}")
    print(f"Type of baseball array: {type(baseball_arr)}")


    height_in_arr = np.array(datasets.height_in)
    print(f"np_height_in=array({height_in_arr}, shape={height_in_arr.shape})")
    height_in_arr = height_in_arr * 0.0254
    print(f"np_height_metres=array({height_in_arr}, shape={height_in_arr.shape})")
if __name__ == '__main__':
    main()
