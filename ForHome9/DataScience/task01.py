import re


test_string = "Let's write RegEx!  Won't that be fun?  I sure think so.  Can you find 4 sentences?  Or perhaps, all 19 words?"


def main():
    sentences = re.findall(r"[^.!?]*[.!?]", test_string)
    print("Sentences:")
    for sentence in sentences:
        print(sentence)

    capitalizedWords = re.findall(r"[A-Z][\w']*", test_string)
    print("Capitalized words:")
    for word in capitalizedWords:
        print(word)

    tokens = re.split("\s", test_string)
    print(f"{tokens=}")

    numbers = re.findall(r"[0-9]+", test_string)
    print(f"{numbers=}")

if __name__ == '__main__':
    main()