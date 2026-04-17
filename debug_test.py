#!/usr/bin/env python3
import traceback
from appworld import AppWorld, load_task_ids

tasks = load_task_ids()  
world = AppWorld(task_id=tasks[0])

# Check if there are any existing users in Spotify
from appworld.apps.spotify.models import User as SpUser

users = SpUser.find_all()
print(f"Existing Spotify users: {len(users) if users else 0}")

if users:
    user = users[0]
    print(f"\nTesting with user: {user.email}")
    
    # Try login
    try:
        print("\nCalling spotify.login()...")
        result = world.apis.spotify.login(username=user.email, password=user.password)
        print(f"✓ Login successful!\nResult: {result}")
    except TypeError as e:
        if "'NoneType' object is not iterable" in str(e):
            print("\n!!! FOUND THE BUG !!!")
            print(f"Error: {e}\n")
            traceback.print_exc()
        else:
            print(f"Different TypeError: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"\nOther error ({type(e).__name__}): {str(e)[:300]}")
        if "NoneType" in str(e):
            traceback.print_exc()
else:
    print("No existing users found. Creating a test user...")
    try:
        result = world.apis.spotify.signup(
            first_name="Test",
            last_name="User",
            email="test_debug@example.com",
            password="Password123!"
        )
        print(f"Signup result: {result}")
        
        # Try login
        result = world.apis.spotify.login(username="test_debug@example.com", password="Password123!")
        print(f"Login result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
