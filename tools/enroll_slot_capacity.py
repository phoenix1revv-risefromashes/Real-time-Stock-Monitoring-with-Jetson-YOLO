from pathlib import Path
import json


REFERENCE_PATH = Path("configs/shelf_base_reference.json")


def main():
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing reference file: {REFERENCE_PATH}. "
            "Run shelf-base enrollment first."
        )

    with open(REFERENCE_PATH, "r") as file:
        reference_data = json.load(file)

    print()
    print("Enroll maximum item capacity for each shelf slot.")
    print("Example: if S1 can hold 12 cans/books, enter 12.")
    print()

    for slot in reference_data["slots"]:
        slot_name = slot["name"]
        existing_capacity = slot.get("max_item_capacity")

        if existing_capacity is None:
            prompt = f"{slot_name} max item capacity: "
        else:
            prompt = f"{slot_name} max item capacity [{existing_capacity}]: "

        user_input = input(prompt).strip()

        if user_input == "" and existing_capacity is not None:
            max_capacity = int(existing_capacity)
        else:
            max_capacity = int(user_input)

        if max_capacity <= 0:
            raise ValueError("Max item capacity must be greater than 0.")

        slot["max_item_capacity"] = max_capacity

    with open(REFERENCE_PATH, "w") as file:
        json.dump(reference_data, file, indent=4)

    print()
    print(f"Saved slot capacities to {REFERENCE_PATH}")


if __name__ == "__main__":
    main()