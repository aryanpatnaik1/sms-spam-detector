totalmessages = 0
totalspam = 0
totalham = 0
unique_messages = {}
conflicting_labels = 0
duplicates = 0

with open("data/SMSSpamCollection", encoding="utf-8") as file:
    for line in file:
        totalmessages += 1
        cleaned_line = line.rstrip("\n")
        parts = cleaned_line.split("\t", 1)
        label = parts[0]
        message = parts[1]

        if label == "spam":
            totalspam += 1
        elif label == "ham":
            totalham += 1
        else:
            raise ValueError(f"Unknown label on line {totalmessages}")

        if message in unique_messages:
            duplicates += 1
            if unique_messages[message] != label:
                conflicting_labels += 1
        else:
            unique_messages[message] = label

print("Total messages:", totalmessages)
print("Total spam:", totalspam)
print("Total ham:", totalham)
print("Duplicates found:", duplicates)
print("Unique messages:", len(unique_messages))
print("Conflicting labels:", conflicting_labels)