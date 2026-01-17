users = {}

def create_user():
    user_id = input("Enter User ID: ")
    name = input("Enter Name: ")
    email = input("Enter Email: ")
    users[user_id] = {"name": name, "email": email}
    print("✅ User created")

def read_user():
    user_id = input("Enter User ID: ")
    user = users.get(user_id)
    if user:
        print("👤 User:", user)
    else:
        print("❌ User not found")

def update_user():
    user_id = input("Enter User ID: ")
    if user_id in users:
        name = input("Enter new name: ")
        email = input("Enter new email: ")
        users[user_id]["name"] = name
        users[user_id]["email"] = email
        print("✏️ User updated")
    else:
        print("❌ User not found")

def delete_user():
    user_id = input("Enter User ID: ")
    if user_id in users:
        del users[user_id]
        print("🗑️ User deleted")
    else:
        print("❌ User not found")

def show_all_users():
    print("\n📋 All Users:")
    for uid, data in users.items():
        print(uid, "=>", data)

# ---- MAIN LOOP ----
while True:
    print("""
1. Create User
2. Read User
3. Update User
4. Delete User
5. Show All Users
6. Exit
""")

    choice = input("Choose option: ")

    if choice == "1":
        create_user()
    elif choice == "2":
        read_user()
    elif choice == "3":
        update_user()
    elif choice == "4":
        delete_user()
    elif choice == "5":
        show_all_users()
    elif choice == "6":
        print("👋 Exiting...")
        break
    else:
        print("❌ Invalid choice")
