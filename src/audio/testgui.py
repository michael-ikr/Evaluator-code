# import mingus.extra.LilyPond as LilyPond

import abjad
def main():
    
    string = "c'16 f' g' a' d' g' a' b' e' a' b' c'' f' b' c'' d''16"
    voice_1 = abjad.Voice(string, name="Voice_1")
    staff_1 = abjad.Staff([voice_1], name="Staff_1")
    abjad.show(staff_1)
    
    
    print(voice_1)
    
     
    
    
if __name__ == '__main__':
    main()