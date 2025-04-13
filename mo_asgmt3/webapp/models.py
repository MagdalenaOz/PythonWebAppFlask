from webapp import db


class DataPoint(db.Model):
    id = db.mapped_column(db.Integer, primary_key=True)
    sepal_length = db.mapped_column(db.Float, nullable=False)
    sepal_width = db.mapped_column(db.Float, nullable=False)
    petal_length = db.mapped_column(db.Float, nullable=False)
    petal_width = db.mapped_column(db.Float, nullable=False)
    species = db.mapped_column(db.Integer, nullable=False)

    def __repr__(self):
        return (f"<DataPoint(id={self.id}, sepal_length={self.sepal_length}, "
                f"sepal_width={self.sepal_width}, petal_length={self.petal_length}, "
                f"petal_width={self.petal_width}, species={self.species})>")
