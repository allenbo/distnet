from distbase.util import isfloat, isinteger

class SettingReader(object):
  @staticmethod
  def read(file):
    rst = {}
    with open(file) as f:
      for line in f:
        line = line.strip()
        if line.startswith('#'):
          continue
        else:
          key = line[0:line.find('=')]
          value = line[line.find('=')+1: len(line)]

          if value.isdigit():
            value = int(value)
          elif isfloat(value):
            value = float(value)
          else:
            value = eval(value)

          rst[key] = value
    return rst

def getmodel(modelfile):
  rst = []
  with open(modelfile) as f:
    for line in f:
      line = line.strip()
      if line.startswith('#'):
        continue
      elif line.startswith('['):
        name = line[1:line.find(']')]
        rst.append({'name':name})
      elif len(line) == 0:
        continue
      else:
        key = line[0:line.find('=')]
        value = line[line.find('=')+1: len(line)]

        if value.isdigit():
          value = int(value)
        elif isfloat(value):
          value = float(value)

        rst[-1][key] = value
  return rst

