
from Listener import Listener
from Calculator import Calculator

def main():
    listener = Listener()
    buffer = listener.listen(duration=5)
    calculator = Calculator()
    df = calculator.calculate(buffer, fast = True, onsets=False, create_file=False, out_file="missed.csv")
    # df = calculator.calculate_and_compare(buffer, create_file=False, out_file="missed.csv")

    print(df)
    #NOTE Right now, yin is converting a rest to a C7!!!!
    #Fix that. C7 is just the lowest possible note.
    
    
# TODO: With fast (yin) audio processing, rests are being detected as C7. Maybe change threshold for rest?

if __name__ == "__main__":
    main()