import json 


class Conf:
    def __init__(self, conf_path="./conf/train.json") -> None:
        self.__dict = {}
        with open(conf_path, "r") as fp:
            self.__dict.update(json.load(fp))
        self.__dict__.update(self.__dict)

    def __len__(self):
        return len(self.__dict)
    
    def __getitem__(self, x):
        return self.__dict[x]

    def __str__(self) -> str:
        ret = ["{}: {}".format(key, value) for key, value in self.__dict.items()]
        return '\n' + '\n'.join(ret)
    

if __name__ == "__main__":
    conf = Conf()
    print(len(conf))
    print(conf["epochs"])
    print(conf.epochs)
    print(conf)
        