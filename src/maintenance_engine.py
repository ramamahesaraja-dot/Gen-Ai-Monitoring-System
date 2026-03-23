def recommend_maintenance(sensor_data):
    actions = []

    if sensor_data.get("vibration_rms", 0) > 4.5:
        actions.append({
            "equipment": "Rotating equipment / bearing / alignment",
            "priority": "HIGH",
            "reason": "Vibration sangat tinggi, indikasi degradasi mekanis",
            "action": "Inspeksi bearing, alignment, balancing, dan mounting"
        })

    if sensor_data.get("motor_current", 0) > 85:
        actions.append({
            "equipment": "Motor drive system",
            "priority": "HIGH",
            "reason": "Arus motor tinggi, indikasi overload atau mechanical resistance",
            "action": "Periksa load motor, wiring, dan mechanical coupling"
        })

    if sensor_data.get("reactor_pressure", 0) > 22:
        actions.append({
            "equipment": "Pressure control loop / valve / relief system",
            "priority": "HIGH",
            "reason": "Tekanan tinggi mendekati kondisi tidak aman",
            "action": "Verifikasi valve, transmitter, controller, dan relief valve"
        })

    if sensor_data.get("efficiency_loss_pct", 0) > 10:
        actions.append({
            "equipment": "Heat transfer system / exchanger surface",
            "priority": "MEDIUM",
            "reason": "Efficiency loss tinggi, kemungkinan fouling atau scaling",
            "action": "Jadwalkan inspeksi permukaan perpindahan panas dan cleaning"
        })

    if not actions:
        actions.append({
            "equipment": "General monitoring",
            "priority": "LOW",
            "reason": "Tidak ada indikasi deviasi signifikan",
            "action": "Lanjutkan routine inspection dan preventive maintenance"
        })

    return actions