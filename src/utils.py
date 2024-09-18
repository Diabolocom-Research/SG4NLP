'''Set of utility function. Primarily redis caching layer'''
import time

def caching_layer(model_name, model_temperature, message, llm, redis_cli,
                  use_redis_caching):
    sleep_time = 1
    if model_temperature != 0.0 or use_redis_caching is False:
        time.sleep(sleep_time)
        output = llm.invoke(message)
        if type(output) != str:
            output = output.content

        return output

    composite_key = str(model_name) + str(model_temperature) + str(message)
    cached_response = redis_cli.get(composite_key)
    if cached_response:
        return cached_response
    else:
        time.sleep(sleep_time)
        output = llm.invoke(message)
        if type(output) != str:
            output = output.content
        redis_cli.set(composite_key, output)
        return output