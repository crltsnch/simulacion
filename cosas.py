from selenium import webdriver
from bs4 import BeautifulSoup

# URL de la página web
url = "https://fbref.com/es/comps/8/1992-1993/stats/Estadisticas-1992-1993-Champions-League"

# Configuración del navegador
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # Para ejecutar el navegador en segundo plano (sin interfaz gráfica)
driver = webdriver.Chrome(options=options)

# Obtener la página web con Selenium
driver.get(url)

# Esperar a que la página cargue completamente (puedes ajustar el tiempo según sea necesario)
driver.implicitly_wait(10)

# Obtener el contenido HTML después de que la página haya cargado
html = driver.page_source

# Cerrar el navegador
driver.quit()

# Utilizar BeautifulSoup para analizar el HTML
soup = BeautifulSoup(html, 'html.parser')

# Encontrar el contenedor con la clase "table_container" y el ID "div_stats_standard"
container = soup.find('div', {'class': 'table_container', 'id': 'div_stats_standard'})

# Verificar si el contenedor se encontró
if container:
    # Encontrar la tabla dentro del contenedor por la clase "stats_table"
    table = container.find('table', {'class': 'stats_table'})

    # Verificar si la tabla se encontró
    if table:
        # Iterar sobre las filas de la tabla
        for row in table.find_all('tr'):
            # Obtener los datos de cada celda en la fila
            cells = row.find_all(['th', 'td'])
            row_data = [cell.text.strip() for cell in cells]

            # Imprimir los datos de la fila
            print(row_data)
    else:
        print("No se encontró la tabla dentro del contenedor.")
else:
    print("No se encontró el contenedor con la clase 'table_container' y el ID 'div_stats_standard'.")