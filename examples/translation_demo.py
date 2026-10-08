from translation.i18n_router import translate_text

def main():
    text = "Bonjour, comment allez-vous ?"

    print("\n===== DONASSOS AI TRANSLATION =====\n")

    print("Original:")
    print(text)

    print("\nEnglish:")
    print(translate_text(text, "fr", "en"))

    print("\nArabic:")
    print(translate_text(text, "fr", "ar"))


if __name__ == "__main__":
    main()