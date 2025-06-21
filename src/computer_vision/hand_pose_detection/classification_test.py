import new_classification_fixed
def main():
    classifier = new_classification_fixed.Classification()
    v = ((10, 50, 150, 55), (10.2, -500, 155, 60))
    # correct
    l = (1.0 / 6.0, 100)
    print(classifier.intersects_vertical(linear_line = l, vertical_lines = v))
    # only intersects right string
    l = (1.0 / 6.0, 53)
    print(classifier.intersects_vertical(linear_line = l, vertical_lines = v))
    # only intersects left string
    l = (1.0 / 3.0, 140)
    print(classifier.intersects_vertical(linear_line = l, vertical_lines = v))
    # too low
    l = (1.0 / 6.0, 60)
    print(classifier.intersects_vertical(linear_line = l, vertical_lines = v))
    # too high
    l = (1.0 / 6.0, 140)
    print(classifier.intersects_vertical(linear_line = l, vertical_lines = v))

if __name__ == "__main__":
    main()