import glob as g
import multiprocessing.pool as pool

dirc = input("images dirctory(FULL PATH)>>")

SIZE_H = int(input("image size height>>"))
SIZE_W = int(input("image size width>>"))
multi = int(input("multi prosess num>>"))


def convertFormat(path: str):
    try:
        f = open(path, "r")

        text = " ".join(f.readlines())
        f.close()
        textlist = text.split()
        textlist = list(map(int, textlist))
        classNumber = textlist[0]
        leftUpX = textlist[1]
        leftUpY = textlist[2]

        rightDownX = textlist[3]
        rightDownY = textlist[4]

        width = (rightDownX - leftUpX) / SIZE_W
        height = (rightDownY - leftUpY) / SIZE_H

        rx = width / 2
        ry = height / 2

        width = "{:.7g}".format(width)
        height = "{:.7g}".format(height)
        rx = "{:.7g}".format(rx)
        ry = "{:.7g}".format(ry)

        result = "{} {} {} {} {}".format(classNumber, rx, ry, width, height)
        f = open(path, "w")
        print(result, file=f)
        f.close()
    except:
        print("Err:file=", path)
        import traceback
        traceback.print_exc()


print("=============== convert labals script !!overWrite!! ===============")
print("=============== written by IDICHI =================================")
print("=============== access https://idichi.tk now!======================")

# collect lists
if dirc[-1] == "/":
    findPath = dirc + "/**/*.txt"
else:
    findPath = dirc + "/**/*.txt"
print("find path", findPath)
txtfiles = g.glob(findPath, recursive=True)
print("find", len(txtfiles), "files!")
print("convert start")
p = pool.Pool(multi)
p.map(convertFormat, txtfiles)

print("sucsuss")
