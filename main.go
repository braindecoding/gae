package main

import (
	"flag"
	"fmt"
	"image/jpeg"
	"log"
	"os"

	_ "net/http/pprof"

	"github.com/aiteung/mnist"
	"gopkg.in/cheggaaa/pb.v1"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"time"
)

func main() {
	flag.Parse()
	parseDtype()

	// // intercept Ctrl+C
	// sigChan := make(chan os.Signal, 1)
	// signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	// doneChan := make(chan bool, 1)

	var inputs tensor.Tensor
	var err error

	// load our data set
	trainOn := *dataset
	if inputs, _, err = mnist.Load(trainOn, loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples := inputs.Shape()[0]
	bs := *batchsize

	// MNIST data consists of 28 by 28 black and white images
	// however we've imported it directly now as 784 different pixels
	// as a result, we need to reshape it to match what we actually want
	// if err := inputs.Reshape(numExamples, 1, 28, 28); err != nil {
	// 	log.Fatal(err)
	// }

	// we should now also proceed to put in our desired variables
	// x is where our input should go, while y is the desired output
	g := gorgonia.NewGraph()
	// x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	x := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("y"))

	os.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)

	m := newNN(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	os.WriteFile("simple_graph_2.dot", []byte(g.ToDot()), 0644)

	losses, err := gorgonia.Square(gorgonia.Must(gorgonia.Sub(y, m.out)))
	if err != nil {
		log.Fatal(err)
	}
	cost := gorgonia.Must(gorgonia.Mean(losses))
	// cost = gorgonia.Must(gorgonia.Neg(cost))

	os.WriteFile("simple_graph_3.dot", []byte(g.ToDot()), 0644)

	// we wanna track costs
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	// logger := log.New(os.Stderr, "", 0)
	// vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...), gorgonia.WithLogger(logger), gorgonia.WithWatchlist(), gorgonia.WithValueFmt("%1.1s"))

	// vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...), gorgonia.WithLogger(logger), gorgonia.TraceExec())
	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewAdamSolver(gorgonia.WithBatchSize(float64(bs)), gorgonia.WithLearnRate(0.01))

	batches := numExamples / bs
	log.Printf("Batches %d", batches)
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)

	for i := 0; i < *epochs; i++ {
		// for i := 0; i < 10; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			// var xVal, yVal tensor.Tensor
			var xVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			// if yVal, err = inputs.Slice(sli{start, end}); err != nil {
			// 	log.Fatal("Unable to slice y")
			// }
			// if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
			// 	log.Fatal("Unable to reshape %v", err)
			// }
			if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
				log.Printf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, xVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}

			arrayOutput := m.predVal.Data().([]float64)
			yOutput := tensor.New(tensor.WithShape(bs, 784), tensor.WithBacking(arrayOutput))

			for j := 0; j < 1; j++ {
				rowT, _ := yOutput.Slice(sli{j, j + 1})
				row := rowT.Data().([]float64)

				img := visualizeRow(row)

				f, _ := os.OpenFile(fmt.Sprintf("training/%d - %d - %d training.jpg", j, b, i), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
				jpeg.Encode(f, img, &jpeg.Options{})
				f.Close()
			}

			// solver.Step(m.learnables())
			solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
			vm.Reset()
			bar.Increment()
		}
		bar.Update()
		log.Printf("Epoch %d | cost %v", i, costVal)
	}
	bar.Finish()

	log.Printf("Run Tests")

	// load our test set
	if inputs, _, err = mnist.Load("test", loc, dt); err != nil {
		log.Fatal(err)
	}

	numExamples = inputs.Shape()[0]
	bs = *batchsize
	batches = numExamples / bs

	bar = pb.New(batches)
	bar.SetRefreshRate(time.Second / 20)
	bar.SetMaxWidth(80)
	bar.Prefix("Epoch Test")
	bar.Set(0)
	bar.Start()
	for b := 0; b < batches; b++ {
		start := b * bs
		end := start + bs
		if start >= numExamples {
			break
		}
		if end > numExamples {
			end = numExamples
		}

		var xVal tensor.Tensor
		if xVal, err = inputs.Slice(sli{start, end}); err != nil {
			log.Fatal("Unable to slice x")
		}

		// if yVal, err = inputs.Slice(sli{start, end}); err != nil {
		// 	log.Fatal("Unable to slice y")
		// }
		// if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
		// 	log.Fatal("Unable to reshape %v", err)
		// }
		if err = xVal.(*tensor.Dense).Reshape(bs, 784); err != nil {
			log.Printf("Unable to reshape %v", err)
		}

		gorgonia.Let(x, xVal)
		gorgonia.Let(y, xVal)
		if err = vm.RunAll(); err != nil {
			log.Printf("Failed at epoch test: %v", err)
		}

		for j := 0; j < xVal.Shape()[0]; j++ {
			rowT, _ := xVal.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			f, _ := os.OpenFile(fmt.Sprintf("images/%d - %d input.jpg", b, j), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			jpeg.Encode(f, img, &jpeg.Options{})
			f.Close()
		}

		arrayOutput := m.predVal.Data().([]float64)
		yOutput := tensor.New(tensor.WithShape(bs, 784), tensor.WithBacking(arrayOutput))

		for j := 0; j < yOutput.Shape()[0]; j++ {
			rowT, _ := yOutput.Slice(sli{j, j + 1})
			row := rowT.Data().([]float64)

			img := visualizeRow(row)

			f, err := os.OpenFile(fmt.Sprintf("images/%d - %d output.jpg", b, j), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
			if err != nil {
				fmt.Printf("\nError terjadi : %v \n", err)
			}

			jpeg.Encode(f, img, &jpeg.Options{})
			f.Close()
		}

		vm.Reset()
		bar.Increment()
	}

	save(m)

	log.Printf("Epoch Test | cost %v", costVal)

}
