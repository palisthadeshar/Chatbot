from functions import *


data = []
main_url = get_url('https://tai.com.np')
filtered_url = filter_url(main_url)
data_list = dataset(filtered_url,data)