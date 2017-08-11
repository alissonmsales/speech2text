from pydub import AudioSegment
import math
sound = AudioSegment.from_file(
    "/media/alissonsales/Files/base_dados/pt_05.mp3")

print(len(sound))

num_pdco = math.floor(len(sound)/9980)
print(num_pdco)
ini = 0
fim = 9980

n = 3820
num_pdco+=n

for x in range(n, num_pdco):
    nome = '{num:05d}_pt.mp3'.format(num=x)
    print(nome)
    file = sound[ini:fim]
    file.export("/media/alissonsales/Files/base_dados/pt_05/"+nome,
                format="mp3")
    ini += 9980


    fim += 9980