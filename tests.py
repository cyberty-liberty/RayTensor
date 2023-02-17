from raytensor import RayTensor


def raytest():
    RayTensor().xray_predict("tests/unittest/xray_test.png")
    RayTensor().ct_predict("tests/unittest/ct_test.png")


if __name__ == "__main__":
    raytest()
