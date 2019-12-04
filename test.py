class Pizza(object):
    radius = 42

    @classmethod
    def get_radius(cls):
        return cls.radius

# Both methods bound to the class, not instance
# <bound method Pizza.get_radius of <class '__main__.Pizza'>>
print(Pizza.get_radius)
print(Pizza().get_radius)
print(Pizza.get_radius == Pizza().get_radius)  # TRUE
# Doesn't need an instance because its bound to the class
print(Pizza.get_radius())  # 42

