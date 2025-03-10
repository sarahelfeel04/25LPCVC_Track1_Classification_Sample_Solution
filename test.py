import qai_hub

available_devices = qai_hub.get_devices()
for device in available_devices:
    print(device)