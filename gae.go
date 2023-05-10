package main

import (
	"encoding/gob"
	"flag"
	"image"
	"log"
	"math"
	"os"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	epochs    = flag.Int("epochs", 5, "Number of epochs to train for")
	dataset   = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype     = flag.String("dtype", "float64", "Which dtype to use")
	batchsize = flag.Int("batchsize", 100, "Batch size")
)

const loc = "./mnist/"
const backup = "./backup/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type nn struct {
	g              *gorgonia.ExprGraph
	w0, w1, w2, w3 *gorgonia.Node

	out     *gorgonia.Node
	predVal gorgonia.Value
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func newNN(g *gorgonia.ExprGraph) *nn {
	// Create node for w/weight
	w0 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(784, 128), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128, 64), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w2 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(64, 128), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	w3 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(128, 784), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))

	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
	}
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2, m.w3}
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var l0, l1, l2, l3, l4 *gorgonia.Node
	var l0dot, l1dot, l2dot, l3dot *gorgonia.Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for Sigmoid
	if l0dot, err = gorgonia.Mul(l0, m.w0); err != nil {
		return errors.Wrap(err, "Unable to multiple l0 and w0")
	}
	l1 = gorgonia.Must(gorgonia.Sigmoid(l0dot))

	if l1dot, err = gorgonia.Mul(l1, m.w1); err != nil {
		return errors.Wrap(err, "Unable to multiple l1 and w1")
	}
	l2 = gorgonia.Must(gorgonia.Sigmoid(l1dot))

	if l2dot, err = gorgonia.Mul(l2, m.w2); err != nil {
		return errors.Wrap(err, "Unable to multiple l2 and w2")
	}
	l3 = gorgonia.Must(gorgonia.Sigmoid(l2dot))

	if l3dot, err = gorgonia.Mul(l3, m.w3); err != nil {
		return errors.Wrap(err, "Unable to multiple l3 and w3")
	}
	l4 = gorgonia.Must(gorgonia.Sigmoid(l3dot))

	// m.pred = l3dot
	// gorgonia.Read(m.pred, &m.predVal)
	// return nil

	m.out = l4
	gorgonia.Read(l4, &m.predVal)
	return

}

const pixelRange = 255

func reversePixelWeight(px float64) byte {
	// return byte((pixelRange*px - pixelRange) / 0.9)
	return byte(pixelRange*math.Min(0.99, math.Max(0.01, px)) - pixelRange)
}

func visualizeRow(x []float64) *image.Gray {
	// since this is a square, we can take advantage of that
	l := len(x)
	side := int(math.Sqrt(float64(l)))
	r := image.Rect(0, 0, side, side)
	img := image.NewGray(r)

	pix := make([]byte, l)
	for i, px := range x {
		pix[i] = reversePixelWeight(px)
	}
	img.Pix = pix

	return img
}

func save(model *nn) error {
	f, err := os.Create(backup + "backup2.gob")
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	for _, node := range model.learnables() {
		err := enc.Encode(node.Value())
		if err != nil {
			return err
		}
	}
	return nil
}
