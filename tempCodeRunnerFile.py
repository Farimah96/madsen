
    # try:
    #     output = subprocess.check_output(cmd, env=env).decode()
    #     return parse_throughput(output)
    # except subprocess.CalledProcessError as e:
    #     print("Error executing sdf3:", e)
    #     print("Output:", e.output.decode() if e.output else "")
    #     return 0.0
