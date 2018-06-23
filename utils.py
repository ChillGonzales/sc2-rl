""" Maps [-1, 1] range to [0, 2] and then converts to z-score"""
def convert_to_zscore(value):
  return (2 - (value + 1)) / 2